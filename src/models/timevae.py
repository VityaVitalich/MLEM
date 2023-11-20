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
    epsilon = torch.randn(k_iwae, mean.shape[0], mean.shape[1]).to(logstd.device)
    z = epsilon * torch.exp(0.5 * logstd) + mean  # modified
    z = z.view(-1, mean.shape[1])
    return z


class TimeVAE(nn.Module):
    def __init__(self, model_conf, data_conf):
        super().__init__()
        self.model_conf = model_conf
        self.data_conf = data_conf

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
        self.out_dim = self.input_dim
        if self.model_conf.time_embedding:
            self.input_dim += self.model_conf.time_embedding * 2 - 1

        ### INIT ENCODER
        ls = []
        prev_dim = self.input_dim
        for i, dim in enumerate(self.model_conf.timevae.hiddens):
            ls.append(
                nn.Conv1d(
                    in_channels=prev_dim,
                    out_channels=dim,
                    kernel_size=3,
                    stride=2,
                    padding=3,
                )
            )
            ls.append(nn.ReLU())
            prev_dim = dim

        ls.append(nn.Flatten())
        self.encoder_conv = nn.Sequential(*ls)
        self.flat_size = self.detect_size(
            self.input_dim, self.data_conf.train.max_seq_len
        )

        self.mu_head = nn.Linear(self.flat_size, self.model_conf.timevae.latent_dim)
        self.log_std_head = nn.Linear(
            self.flat_size, self.model_conf.timevae.latent_dim
        )

        ### Decoder Init ###
        self.dec_proj = nn.Linear(self.model_conf.timevae.latent_dim, self.flat_size)

        ls = []
        prev_dim = self.model_conf.timevae.hiddens[-1]
        for i, dim in enumerate(reversed(self.model_conf.timevae.hiddens[:-1])):
            ls.append(
                nn.ConvTranspose1d(
                    in_channels=prev_dim,
                    out_channels=dim,
                    kernel_size=3,
                    stride=2,
                    padding=2,
                )
            )
            ls.append(nn.ReLU())
            prev_dim = dim
        ls.append(
            nn.ConvTranspose1d(
                in_channels=prev_dim,
                out_channels=self.input_dim,
                kernel_size=3,
                stride=2,
                padding=2,
            )
        )
        ls.append(nn.ReLU())
        ls.append(nn.Flatten())
        self.decoder = nn.Sequential(*ls)
        out_proj_size = self.detect_dec_size()
        self.decoder_out_proj = nn.Linear(
            out_proj_size, self.data_conf.train.max_seq_len * self.out_dim
        )

        ### LOSS ###
        self.embedding_predictor = EmbeddingPredictor(
            model_conf=self.model_conf, data_conf=self.data_conf
        )
        self.numeric_projector = NumericalFeatureProjector(
            model_conf=self.model_conf, data_conf=self.data_conf
        )
        self.mse_fn = torch.nn.MSELoss(reduction="none")
        self.ce_fn = torch.nn.CrossEntropyLoss(
            reduction="mean", ignore_index=0, label_smoothing=0.05
        )

    def forward(self, padded_batch):
        x, time_steps = self.processor(padded_batch)
        x = self.time_encoder(x, time_steps)

        mu, log_std = self.encode(x.transpose(1, 2))
        z = sample_z(mu, log_std, 1)
        pred = self.decode(z)

        gt = {"input_batch": padded_batch, "time_steps": time_steps}

        res_dict = {"gt": gt, "pred": pred, "latent": mu, "mu": mu, "log_std": log_std}
        return res_dict

    def decode(self, z):
        projected = self.dec_proj(z).view(
            z.size(0), self.model_conf.timevae.hiddens[-1], -1
        )
        decoded = self.decoder(projected)
        out = self.decoder_out_proj(decoded)
        out = out.view(z.size(0), self.data_conf.train.max_seq_len, self.out_dim)

        pred = self.embedding_predictor(out)
        pred.update(self.numeric_projector(out))

        if self.model_conf.use_deltas:
            pred["delta"] = out[:, :, -1].squeeze(-1)

        return pred

    def encode(self, x):
        features = self.encoder_conv(x)
        mu = self.mu_head(features)
        log_std = self.log_std_head(features)

        return mu, log_std

    def detect_size(self, in_dim, seq_len):
        test_value = torch.rand(1, seq_len, in_dim)

        with torch.no_grad():
            out = self.encoder_conv(test_value.transpose(1, 2))

        return out.size(1)

    def detect_dec_size(self):
        test_value = torch.rand(1, self.model_conf.timevae.latent_dim)
        with torch.no_grad():
            projected = self.dec_proj(test_value).view(
                1, self.model_conf.timevae.hiddens[-1], -1
            )
            decoded = self.decoder(projected)

        return decoded.size(1)

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
            "total_KL_loss": kl_loss,
            "delta_loss": delta_mse_loss,
        }
        losses_dict.update(cross_entropy_losses)

        total_loss = (
            self.model_conf.mse_weight * losses_dict["total_mse_loss"]
            + self.model_conf.CE_weight * total_ce_loss
            + self.model_conf.delta_weight * delta_mse_loss
        ) * self.model_conf.timevae.recon_weight + kl_loss
        losses_dict["total_loss"] = total_loss

        return losses_dict

    def generate(self, padded_batch, lens):
        bs = padded_batch.payload["event_time"].size(0)
        Z = torch.randn(bs, self.model_conf.timevae.latent_dim).to(
            self.model_conf.device
        )
        samples = self.decode(Z)
        return {"pred": samples}

    def numerical_loss(self, output):
        # MSE
        total_mse_loss = 0
        for key, values in output["gt"]["input_batch"].payload.items():
            if key in self.processor.numeric_names:
                gt_val = values.float()
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
