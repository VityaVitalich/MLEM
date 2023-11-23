import torch
import torch.nn as nn
from . import preprocessors as prp
from ..trainers.losses import get_loss
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from .model_utils import get_normal_kl
from functools import partial
from einops import repeat
from .model_utils import NumericalFeatureProjector, EmbeddingPredictor


def sample_z(mean, logstd, k_iwae):
    epsilon = torch.randn(k_iwae, mean.shape[0], mean.shape[1], mean.shape[2]).to(
        logstd.device
    )
    z = epsilon * torch.exp(0.5 * logstd) + mean  # modified
    z = z.view(-1, mean.shape[1], mean.shape[2])
    return z


class TPPVAE(nn.Module):
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
            hidden_size=self.model_conf.tppvae.hidden_rnn,
            num_layers=self.model_conf.tppvae.num_layers_enc,
            batch_first=True,
        )

        self.h0 = nn.Parameter(torch.rand(self.model_conf.tppvae.hidden_rnn))
        ### Encoder ###
        self.net_joint = []
        for i in range(self.model_conf.tppvae.joint_layer_num):
            self.net_joint.append(
                nn.Linear(
                    self.model_conf.tppvae.hidden_rnn, self.model_conf.tppvae.hidden_rnn
                )
            )
            self.net_joint.append(nn.GELU())
        self.encoder_net_joint = nn.Sequential(*self.net_joint)

        self.encoder_net_h = nn.Linear(
            self.model_conf.tppvae.hidden_rnn, self.model_conf.tppvae.hidden_rnn
        )
        self.encoder_net_emb = nn.Linear(
            self.input_dim, self.model_conf.tppvae.hidden_rnn
        )
        self.mu_head = nn.Linear(
            self.model_conf.tppvae.hidden_rnn, self.model_conf.tppvae.hidden_rnn
        )
        self.logstd_head = nn.Linear(
            self.model_conf.tppvae.hidden_rnn, self.model_conf.tppvae.hidden_rnn
        )

        ### Decoder ###
        self.net_joint = []
        for i in range(self.model_conf.tppvae.joint_layer_num):
            self.net_joint.append(
                nn.Linear(
                    self.model_conf.tppvae.hidden_rnn, self.model_conf.tppvae.hidden_rnn
                )
            )
            self.net_joint.append(nn.GELU())
        self.decoder_net_joint = nn.Sequential(*self.net_joint)
        self.decoder_net_h = nn.Linear(
            self.model_conf.tppvae.hidden_rnn, self.model_conf.tppvae.hidden_rnn
        )
        self.decoder_z_emb = nn.Linear(
            self.model_conf.tppvae.hidden_rnn, self.model_conf.tppvae.hidden_rnn
        )

        # predict only next time
        self.delta_head = nn.Linear(self.model_conf.tppvae.hidden_rnn, 1)
        # predict embedding from history
        self.embedding_head = nn.Sequential(
            nn.Linear(
                self.model_conf.tppvae.hidden_rnn, self.model_conf.tppvae.hidden_rnn
            ),
            nn.GELU(),
            nn.Linear(self.model_conf.tppvae.hidden_rnn, self.input_dim),
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

    def delta_mse_loss(self, output):
        # DELTA MSE
        if self.model_conf.use_deltas:
            gt_delta = output["gt"]["time_steps"].diff(1)
            if self.model_conf.use_log_delta:
                gt_delta = torch.log(gt_delta + 1e-15)
            delta_mse = self.mse_fn(gt_delta, output["pred"]["delta"][:, :-1])
            # print(delta_mse, gt_delta[0], output["gt"]["time_steps"].diff(1)[0], output["gt"]["time_steps"][0])
            mask = output["gt"]["time_steps"] != -1

            delta_masked = delta_mse * mask[:, :-1]
            delta_mse = delta_masked.sum() / (mask != 0).sum()
        else:
            delta_mse = torch.tensor(0)

        return delta_mse

    def loss(self, output, ground_truth):
        """
        output: Dict that is outputed from forward method
        """
        ### MSE ###
        total_mse_loss = self.numerical_loss(output)
        delta_mse_loss = self.delta_mse_loss(output)

        ### CROSS ENTROPY ###
        cross_entropy_losses = self.embedding_predictor.loss(
            output["pred"], output["gt"]["input_batch"]
        )
        total_ce_loss = torch.sum(
            torch.cat([value.unsqueeze(0) for _, value in cross_entropy_losses.items()])
        )

        kl_loss = (
            get_normal_kl(mean_1=output["mu"], log_std_1=output["log_std"])
            .sum(dim=1)
            .mean()
        )

        losses_dict = {
            "total_mse_loss": total_mse_loss,
            "total_CE_loss": total_ce_loss,
            "delta_loss": self.model_conf.delta_weight * delta_mse_loss,
            "kl_loss": kl_loss,
        }
        losses_dict.update(cross_entropy_losses)

        total_loss = (
            self.model_conf.mse_weight * losses_dict["total_mse_loss"]
            + self.model_conf.CE_weight * total_ce_loss
            + self.model_conf.delta_weight * delta_mse_loss
            + kl_loss
        )
        losses_dict["total_loss"] = total_loss

        return losses_dict

    def forward(self, padded_batch):
        x, time_steps = self.processor(padded_batch)
        x = self.time_encoder(x, time_steps)

        history_emb, mu, log_std = self.encode(x)
        z = sample_z(mu, log_std, k_iwae=1).to(self.model_conf.device)

        pred = self.decode(z, history_emb)

        lens = padded_batch.seq_lens - 1
        global_hidden = history_emb[:, lens, :].diagonal().T

        gt = {"input_batch": padded_batch, "time_steps": time_steps}

        res_dict = {
            "gt": gt,
            "pred": pred,
            "latent": global_hidden,
            "mu": mu,
            "log_std": log_std,
        }
        return res_dict

    def encode(self, x):
        bs, seq_len, dim = x.size()
        history_emb, _ = self.history_encoder(x)
        history_emb = torch.cat(
            [repeat(self.h0, "D -> B L D", B=bs, L=1), history_emb], dim=1
        )[:, :-1, :]

        # use previous history embedding
        out = self.encoder_net_joint(
            self.encoder_net_h(history_emb) + self.encoder_net_emb(x)
        )

        mu = self.mu_head(out)
        log_std = self.logstd_head(out)

        return history_emb, mu, log_std

    def decode(self, z, h):
        # z is sampled but h goes from previous step
        out = self.decoder_net_joint(self.decoder_net_h(h) + self.decoder_z_emb(z))
        pred_delta = self.delta_head(out)
        out = self.embedding_head(h)

        pred = self.embedding_predictor(out)
        pred.update(self.numeric_projector(out))
        pred["delta"] = pred_delta.squeeze(-1)

        return pred

    def generate(self, padded_batch, lens):
        bs, l = padded_batch.payload["event_time"].size()

        initial_state = repeat(self.h0, "D -> BS D", BS=bs)
        z = torch.randn(bs, self.model_conf.tppvae.hidden_rnn).to(
            self.model_conf.device
        )
        out = self.decoder_net_joint(
            self.decoder_net_h(initial_state) + self.decoder_z_emb(z)
        )
        pred_delta = self.delta_head(out).squeeze(-1)
        out = self.embedding_head(initial_state)
        out[:, -1] = pred_delta

        gen_x = torch.zeros(bs, lens, self.input_dim, device=self.model_conf.device)
        gen_x[:, 0, :] = out
        for i in range(1, lens):
            history_emb, _ = self.history_encoder(gen_x)
            history_emb = history_emb[:, i - 1, :]
            z = torch.randn(bs, self.model_conf.tppvae.hidden_rnn).to(
                self.model_conf.device
            )

            out = self.decoder_net_joint(
                self.decoder_net_h(history_emb) + self.decoder_z_emb(z)
            )
            pred_delta = self.delta_head(out).squeeze(-1)
            out = self.embedding_head(history_emb)
            out[:, -1] = pred_delta

            gen_x[:, i, :] = out

        pred = self.embedding_predictor(gen_x)
        pred.update(self.numeric_projector(gen_x))
        pred["delta"] = gen_x[:, :, -1]
        return {"pred": pred}
