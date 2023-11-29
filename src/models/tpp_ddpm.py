import torch
import torch.nn as nn
from . import preprocessors as prp
from ..trainers.losses import get_loss
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from einops import repeat
from .model_utils import NumericalFeatureProjector, EmbeddingPredictor
from functools import partial
from inspect import isfunction


class TPPDDPM(nn.Module):
    def __init__(self, model_conf, data_conf):
        super().__init__()

        self.model_conf = model_conf
        self.data_conf = data_conf

        ### PROCESSORS ###
        self.processor = prp.FeatureProcessor(
            model_conf=model_conf, data_conf=data_conf
        )
        self.time_encoder = prp.TimeEncoder(
            model_conf=self.model_conf, data_conf=self.data_conf
        )

        ### INPUT SIZE ###
        all_emb_size = self.model_conf.features_emb_dim * len(
            self.data_conf.features.embeddings
        )

        self.all_numeric_size = (
            len(self.data_conf.features.numeric_values)
            * self.model_conf.numeric_emb_size
        )

        self.input_dim = (
            all_emb_size + self.all_numeric_size + self.model_conf.use_deltas
        )
        assert self.model_conf.time_embedding == 0
        assert self.model_conf.use_deltas == True

        self.history_encoder = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.model_conf.tppddpm.hidden_rnn,
            num_layers=self.model_conf.tppddpm.num_layers_enc,
            batch_first=True,
        )

        self.h0 = nn.Parameter(torch.rand(self.model_conf.tppddpm.hidden_rnn))

        ### Decoder ###
        self.denoise_net = DenoiseNet(
            self.model_conf.tppddpm.hidden_rnn,
            layer_num=self.model_conf.tppddpm.denoise_layer_num,
            diff_steps=self.model_conf.tppddpm.diff_steps,
        )
        self.diffusion = GaussianDiffusion(
            self.denoise_net, diff_steps=self.model_conf.tppddpm.diff_steps
        )
        # predict embedding from history
        self.embedding_head = nn.Sequential(
            nn.Linear(
                self.model_conf.tppddpm.hidden_rnn, self.model_conf.tppddpm.hidden_rnn
            ),
            nn.GELU(),
            nn.Linear(self.model_conf.tppddpm.hidden_rnn, self.input_dim),
        )

        # Predictors
        self.embedding_predictor = EmbeddingPredictor(
            model_conf=self.model_conf, data_conf=self.data_conf
        )
        self.numeric_projector = NumericalFeatureProjector(
            model_conf=self.model_conf, data_conf=self.data_conf
        )
        self.mse_fn = torch.nn.MSELoss(reduction="none")

    def numerical_loss(self, output):
        # MSE
        total_mse_loss = 0
        for key, values in output["gt"]["input_batch"].payload.items():
            if key in self.processor.numeric_names:
                gt_val = values.float()
                pred_val = output["pred"][key].squeeze(-1)

                mse_loss = self.mse_fn(
                    gt_val,
                    pred_val,
                )
                mask = gt_val != 0
                masked_mse = mse_loss * mask
                total_mse_loss += (
                    masked_mse.sum(dim=1)  # / (mask != 0).sum(dim=1)
                ).mean()

        return total_mse_loss

    def delta_diff_loss(self, output):
        # DELTA MSE

        gt_delta = output["gt"]["time_steps"].diff(1)
        h = output["all_latents"][:, :-1, :]
        log_prob = self.diffusion.log_prob(gt_delta, cond=h)

        return log_prob

    def loss(self, output, ground_truth):
        """
        output: Dict that is outputed from forward method
        """
        ### MSE ###
        total_mse_loss = self.numerical_loss(output)
        delta_diff_loss = self.delta_diff_loss(output)

        ### CROSS ENTROPY ###
        cross_entropy_losses = self.embedding_predictor.loss(
            output["pred"], output["gt"]["input_batch"]
        )
        total_ce_loss = torch.sum(
            torch.cat([value.unsqueeze(0) for _, value in cross_entropy_losses.items()])
        )

        losses_dict = {
            "total_mse_loss": total_mse_loss,
            "total_CE_loss": total_ce_loss,
            "delta_loss": self.model_conf.delta_weight * delta_diff_loss,
        }
        losses_dict.update(cross_entropy_losses)

        total_loss = (
            self.model_conf.mse_weight * losses_dict["total_mse_loss"]
            + self.model_conf.CE_weight * total_ce_loss
            + self.model_conf.delta_weight * delta_diff_loss
        )
        losses_dict["total_loss"] = total_loss

        return losses_dict

    def forward(self, padded_batch, need_delta=False):
        x, time_steps = self.processor(padded_batch)
        x = self.time_encoder(x, time_steps)

        history_emb = self.encode(x)

        pred = self.decode(history_emb, need_delta)

        lens = padded_batch.seq_lens - 1
        global_hidden = history_emb[:, lens, :].diagonal().T

        gt = {"input_batch": padded_batch, "time_steps": time_steps}

        res_dict = {
            "gt": gt,
            "pred": pred,
            "latent": global_hidden,
            "all_latents": history_emb,
        }
        return res_dict

    def encode(self, x):
        bs, seq_len, dim = x.size()
        history_emb, _ = self.history_encoder(x)
        history_emb = torch.cat(
            [repeat(self.h0, "D -> B L D", B=bs, L=1), history_emb], dim=1
        )[
            :, :-1, :
        ]  # shift history emb

        return history_emb

    def decode(self, h, need_delta=False):
        out = self.embedding_head(h)

        pred = self.embedding_predictor(out)
        pred.update(self.numeric_projector(out))

        # need in reconstruction measure
        if need_delta:
            bs, l, d = h.size()
            pred_delta = self.diffusion.sample((bs, l, 1), cond=h)
            pred["delta"] = pred_delta.squeeze(-1)

        return pred

    def generate(self, padded_batch, lens):
        bs, l = padded_batch.payload["event_time"].size()

        initial_state = repeat(self.h0, "D -> BS D", BS=bs)
        out = self.embedding_head(initial_state)
        pred_delta = self.diffusion.sample((bs, 1), cond=initial_state).squeeze(-1)
        out[:, -1] = pred_delta

        gen_x = torch.zeros(bs, lens, self.input_dim, device=self.model_conf.device)
        gen_x[:, 0, :] = out
        for i in range(1, lens):
            history_emb, _ = self.history_encoder(gen_x)
            history_emb = history_emb[:, i - 1, :]
            out = self.embedding_head(history_emb)
            pred_delta = self.diffusion.sample((bs, 1), cond=history_emb).squeeze(-1)
            out[:, -1] = pred_delta

            gen_x[:, i, :] = out

        pred = self.embedding_predictor(gen_x)
        pred.update(self.numeric_projector(gen_x))
        pred["delta"] = gen_x[:, :, -1]
        return {"pred": pred}


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        beta_end=0.1,
        diff_steps=1000,
        loss_type="l2",
        betas=None,
        beta_schedule="linear",
        *args,
        **kwargs
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.input_size = 1
        self.__scale = None

        if betas is not None:
            betas = (
                betas.detach().cpu().numpy()
                if isinstance(betas, torch.Tensor)
                else betas
            )
        else:
            if beta_schedule == "linear":
                betas = np.linspace(1e-4, beta_end, diff_steps)
            elif beta_schedule == "quad":
                betas = np.linspace(1e-4**0.5, beta_end**0.5, diff_steps) ** 2
            elif beta_schedule == "const":
                betas = beta_end * np.ones(diff_steps)
            elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
                betas = 1.0 / np.linspace(diff_steps, 1, diff_steps)
            elif beta_schedule == "sigmoid":
                betas = np.linspace(-6, 6, diff_steps)
                betas = (beta_end - 1e-4) / (np.exp(-betas) + 1) + 1e-4
            elif beta_schedule == "cosine":
                betas = cosine_beta_schedule(diff_steps)
            else:
                raise NotImplementedError(beta_schedule)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, scale):
        self.__scale = scale

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.denoise_fn(x, t, cond=cond)
        )

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, clip_denoised=False, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, cond=cond, t=t, clip_denoised=clip_denoised
        )
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def get_param(self, x, cond, t, clip_denoised=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x,
            cond=cond,
            t=torch.full((b,), t, device=device, dtype=torch.long),
            clip_denoised=clip_denoised,
        )

        return model_mean, (0.5 * model_log_variance).exp()

    def conditional_nll_param(self, shape, cond):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in reversed(range(1, self.num_timesteps)):
            img = self.p_sample(
                img, cond, torch.full((b,), i, device=device, dtype=torch.long)
            )
        gaussian_mu, gaussian_std = self.get_param(img, cond, t=1)
        return gaussian_mu, gaussian_std

    @torch.no_grad()
    def p_sample_loop(self, shape, cond):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)
        # emb = []
        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(
                img, cond, torch.full((b,), i, device=device, dtype=torch.long)
            )

            # emb.append(img)
        # np.save('diffusion_dynamics', torch.stack(emb, dim=0).cpu().numpy())
        return img

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size(), cond=None):
        # if cond is not None:
        #     shape = cond.shape[:-1] + (self.input_size,)
        # else:
        shape = sample_shape
        x_hat = self.p_sample_loop(shape, cond)

        if self.scale is not None:
            x_hat *= self.scale
        return x_hat

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in reversed(range(0, t)):
            img = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long)
            )

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, cond, t, mask=None, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t, cond=cond)

        if mask is not None:
            x_noisy = x_noisy * mask
            x_recon = x_recon * mask

        if self.loss_type == "l1":
            loss = torch.abs(x_recon - noise).sum()
        elif self.loss_type == "l2":
            loss = torch.square(x_recon - noise).sum()
        # elif self.loss_type == "huber":
        #     loss = F.smooth_l1_loss(x_recon, noise)
        else:
            raise NotImplementedError()

        return loss

    def log_prob(self, x, cond, mask=None, *args, **kwargs):
        if self.scale is not None:
            x /= self.scale

        # T = length of sequence
        B, T, D = cond.shape

        time = torch.randint(0, self.num_timesteps, (B * T,), device=x.device).long()
        loss = self.p_losses(
            x.reshape(B * T, -1), cond.reshape(B * T, -1), time
        )  # , mask.reshape(B * T, -1, 1), *args, **kwargs
        #  )

        return loss


class DiffusionEmbedding(nn.Module):
    def __init__(self, embed_size, max_steps=500):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(embed_size, max_steps), persistent=False
        )
        self.projection1 = nn.Linear(embed_size * 2, embed_size)
        self.projection2 = nn.Linear(embed_size, embed_size)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, dim, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(dim).unsqueeze(0)  # [1,dim]
        table = steps * 10.0 ** (dims * 4.0 / dim)  # [T,dim]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class TrigonoTimeEmbedding(nn.Module):
    def __init__(self, embed_size, **kwargs):
        super().__init__()
        assert embed_size % 2 == 0

        self.Wt = nn.Linear(1, embed_size // 2, bias=False)

    def forward(self, interval):
        phi = self.Wt(interval.unsqueeze(-1))
        pe_sin = torch.sin(phi)
        pe_cos = torch.cos(phi)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)
        return pe


class DenoiseNet(nn.Module):
    def __init__(self, embed_size, layer_num, diff_steps, *args, **kwargs):
        super().__init__()
        self.embed_size = embed_size

        self.time_emb = TrigonoTimeEmbedding(embed_size=embed_size)
        self.h_emb = nn.Linear(embed_size, embed_size)
        self.feed_forward = nn.ModuleList(
            [nn.Linear(embed_size, embed_size) for i in range(layer_num)]
        )
        self.to_time = nn.Linear(embed_size, 1)
        self.activation = nn.GELU()
        self.diffusion_time_emb = DiffusionEmbedding(
            embed_size=embed_size, max_steps=diff_steps + 1
        )

    # def forward(self, x, t, cond):
    #     time_embedding = self.time_emb(x)/np.sqrt(self.embed_size) # removed x.squeeze(dim=-1
    #     cond = self.h_emb(cond)
    #     print(time_embedding.size())
    #     b, l, d = time_embedding.shape # l = 1 due to reshape

    #     diff_time_embedding = self.diffusion_time_emb(t)\
    #                           .reshape(b, 1, self.embed_size)\
    #                           .expand_as(time_embedding)

    #     y = time_embedding + diff_time_embedding + cond
    #     for layer in self.feed_forward:
    #         y = layer(y)
    #         y = self.activation(y) + time_embedding + diff_time_embedding + cond
    #     return self.to_time(y)

    def forward(self, x, t, cond):
        time_embedding = self.time_emb(x.squeeze(dim=-1)) / np.sqrt(self.embed_size)
        cond = self.h_emb(cond)
        b, *_ = time_embedding.shape

        diff_time_embedding = (
            self.diffusion_time_emb(t)
            .reshape(b, *(1,) * (len(time_embedding.shape) - 2), self.embed_size)
            .expand_as(time_embedding)
        )

        y = time_embedding + diff_time_embedding + cond
        for layer in self.feed_forward:
            y = layer(y)
            y = self.activation(y) + time_embedding + diff_time_embedding + cond
        return self.to_time(y)
