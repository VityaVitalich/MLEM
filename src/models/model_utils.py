import math

import numpy as np
import scipy
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..data_load.dataloader import PaddedBatch


class EmbeddingPredictor(nn.Module):
    def __init__(self, model_conf, data_conf):
        super().__init__()
        self.model_conf = model_conf
        self.data_conf = data_conf

        self.criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=0)

        self.emb_names = list(self.data_conf.features.embeddings.keys())
        self.num_embeds = len(self.emb_names)
        self.categorical_len = self.num_embeds * self.model_conf.features_emb_dim

        self.init_embed_predictors()

    def init_embed_predictors(self):
        self.embed_predictors = nn.ModuleDict()

        for name in self.emb_names:
            vocab_size = self.data_conf.features.embeddings[name]["max_value"]
            self.embed_predictors[name] = nn.Linear(
                self.model_conf.features_emb_dim, vocab_size
            )

    def forward(self, x_recon):
        batch_size, seq_len, out_dim = x_recon.size()
        resized_x = x_recon[:, :, : self.categorical_len].view(
            batch_size,
            seq_len,
            self.num_embeds,
            self.model_conf.features_emb_dim,
        )
        embeddings_distribution = {}
        for i, name in enumerate(self.emb_names):
            embeddings_distribution[name] = self.embed_predictors[name](
                resized_x[:, :, i, :]
            )

        return embeddings_distribution

    def loss(self, embedding_distribution, padded_batch):
        embed_losses = {}
        for name, dist in embedding_distribution.items():
            if name in self.emb_names:
                shifted_labels = padded_batch.payload[name].long()  # [:, 1:]
                embed_losses[name] = (
                    self.criterion(dist.permute(0, 2, 1), shifted_labels)
                    .sum(dim=1) # changed to mean
                    .mean()
                )

        return embed_losses


class NumericalFeatureProjector(nn.Module):
    def __init__(self, model_conf, data_conf):
        super().__init__()
        self.model_conf = model_conf
        self.data_conf = data_conf

        self.numerical_names = list(self.data_conf.features.numeric_values.keys())
        self.num_embeds = len(self.numerical_names)
        self.numerical_len = (
            self.num_embeds * self.model_conf.numeric_emb_size
            + self.model_conf.use_deltas
        )

        self.init_embed_predictors()

    def init_embed_predictors(self):
        self.embed_predictors = nn.ModuleDict()

        for name in self.numerical_names:
            self.embed_predictors[name] = nn.Linear(self.model_conf.numeric_emb_size, 1)

    def forward(self, x_recon):
        batch_size, seq_len, out_dim = x_recon.size()

        if self.model_conf.use_deltas:
            resized_x = x_recon[:, :, -self.numerical_len : -1]
        else:
            resized_x = x_recon[:, :, -self.numerical_len :]
        # print(resized_x.size())
        resized_x = resized_x.view(
            batch_size,
            seq_len,
            self.num_embeds,
            self.model_conf.numeric_emb_size,
        )

        pred_numeric = {}
        for i, name in enumerate(self.numerical_names):
            pred_numeric[name] = self.embed_predictors[name](resized_x[:, :, i, :])

        return pred_numeric


def calc_intrinsic_dimension(train_embeddings, other_embeddings):
    all_embeddings = []
    train_embeddings = torch.cat(train_embeddings).cpu()
    for other_embedding in other_embeddings:
        if other_embedding is not None:
            all_embeddings.append(torch.cat(other_embedding).cpu())

    X = torch.cat(all_embeddings, dim=0)[-50000:].numpy()

    N = X.shape[0]

    dist = scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(X, metric="euclidean")
    )

    # FOR EACH POINT, COMPUTE mu_i = r_2 / r_1,
    # where r_1 and r_2 are first and second shortest distances
    mu = np.zeros(N)

    for i in range(N):
        sort_idx = np.argsort(dist[i, :])
        mu[i] = dist[i, sort_idx[2]] / (dist[i, sort_idx[1]] + 1e-15)

    # COMPUTE EMPIRICAL CUMULATE
    sort_idx = np.argsort(mu)
    Femp = np.arange(N) / N

    # FIT (log(mu_i), -log(1-F(mu_i))) WITH A STRAIGHT LINE THROUGH ORIGIN
    lr = LinearRegression(fit_intercept=False)
    features = np.log(mu[sort_idx]).reshape(-1, 1)
    features = np.clip(features, 1e-15, 1e15)
    lr.fit(features, -np.log(1 - Femp).reshape(-1, 1))

    d = lr.coef_[0][0]  # extract slope

    return d


def calc_anisotropy(train_embeddings, other_embeddings):
    all_embeddings = []
    train_embeddings = torch.cat(train_embeddings).cpu()
    for other_embedding in other_embeddings:
        if other_embedding is not None:
            all_embeddings.append(torch.cat(other_embedding).cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)[-50000:]

    U, S, Vt = torch.linalg.svd(all_embeddings, full_matrices=False)

    return S[0] / S.sum()


def set_grad(layers, flag):
    for layer in layers:
        for parameter in layer.parameters():
            parameter.requires_grad_(flag)


class L2Normalization(nn.Module):
    def __init__(self):
        super(L2Normalization, self).__init__()

    def forward(self, x):
        return x.div(torch.norm(x, dim=1).view(-1, 1))


class FeatureMixer(nn.Module):
    def __init__(self, num_features, feature_dim, num_layers):
        super().__init__()
        self.num_features = num_features
        self.feature_dim = feature_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=1, batch_first=True
        )
        self.encoder_norm = nn.LayerNorm(feature_dim)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=self.encoder_norm
        )

    def forward(self, x):
        bs, seq_len, d = x.size()

        x_resized = x.view(bs * seq_len, self.num_features, self.feature_dim)
        # x_resized.requires_grad_(True)
        out = self.encoder(x_resized)
        out = out.view(bs, seq_len, self.num_features * self.feature_dim)
        # out.requires_grad_(True)
        return out


def out_to_padded_batch(out, data_conf):
    order = {}

    k = 0
    for key in out["gt"]["input_batch"].payload.keys():
        if key in data_conf.features.numeric_values.keys():
            order[k] = key
            k += 1

    payload = {}
    payload["event_time"] = out["gt"]["time_steps"]
    length = (out["gt"]["time_steps"] != -1).sum(dim=1)
    mask = out["gt"]["time_steps"] != -1
    for key, val in out["pred"].items():
        if key in data_conf.features.embeddings.keys():
            payload[key] = val.cpu().argmax(dim=-1)
            payload[key][~mask] = 0
        elif key in data_conf.features.numeric_values.keys():
            payload[key] = val.cpu().squeeze(-1)
            payload[key][~mask] = 0

    return PaddedBatch(payload, length)


def sample_z(mean, logstd, k_iwae):
    epsilon = torch.randn(k_iwae, mean.shape[0], mean.shape[1], mean.shape[2]).to(
        logstd.device
    )
    z = epsilon * torch.exp(0.5 * logstd) + mean  # modified
    z = z.view(-1, mean.shape[1], mean.shape[2])
    return z


def get_normal_kl(mean_1, log_std_1, mean_2=None, log_std_2=None):
    """
    This function should return the value of KL(p1 || p2),
    where p1 = Normal(mean_1, exp(log_std_1)), p2 = Normal(mean_2, exp(log_std_2) ** 2).
    If mean_2 and log_std_2 are None values, we will use standard normal distribution.
    Note that we consider the case of diagonal covariance matrix.
    """
    if mean_2 is None:
        mean_2 = torch.zeros_like(mean_1).to(mean_1.device)
    if log_std_2 is None:
        log_std_2 = torch.zeros_like(log_std_1).to(mean_1.device)
    # ====
    # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    # https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians

    sigma_1 = torch.exp(log_std_1)
    sigma_2 = torch.exp(log_std_2)

    out = torch.log(sigma_2 / sigma_1)
    out += (sigma_1**2 + (mean_1 - mean_2) ** 2) / (2 * (sigma_2**2))
    out -= 1 / 2

    return out


def get_normal_nll(x, mean, log_std):
    """
    This function should return the negative log likelihood log p(x),
    where p(x) = Normal(x | mean, exp(log_std) ** 2).
    Note that we consider the case of diagonal covariance matrix.
    """
    # ====
    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Likelihood_function
    sigma = torch.exp(log_std.float()).to(x.device)

    out = (((x - mean) / sigma) ** 2) / 2
    out += log_std.to(x.device)
    out += torch.log(torch.sqrt(torch.tensor(2 * np.pi)))

    return out


class MultiTimeAttention(nn.Module):
    def __init__(
        self, input_dim, nhidden=16, embed_time=16, num_heads=1, num_time_emb=1
    ):
        super().__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.num_time_emb = num_time_emb
        self.linears = nn.ModuleList(
            [
                nn.Linear(embed_time, embed_time),
                nn.Linear(embed_time, embed_time),
                nn.Linear(input_dim * num_heads, nhidden),
            ]
        )

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        dim = value.size(-1)
        d_k = query.size(-1)
        # transposed so time embedding dimension of ref points matches time steps
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # repeated to match dimension of input values
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        # softmax over input time points
        p_attn = F.softmax(scores, dim=-2)
        if dropout is not None:
            p_attn = dropout(p_attn)

        # p_attn should be:
        #       (bs, num_time_emb, num_heads, ref_points, time_steps, input_dim)
        # sum over time steps dimension
        return (p_attn * value.unsqueeze(1).unsqueeze(1)).sum(-2), p_attn

    def forward(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, _, dim = value.size()
        num_ref_points = query.size()[2]
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)

        query, key = [
            # divided to  bs, num_time_emb, num_heads, L, dim_head after transpose
            linear(x)
            .view(x.size(0), self.num_time_emb, -1, self.h, self.embed_time_k)
            .transpose(2, 3)
            for linear, x in (zip(self.linears, (query, key)))
        ]
        x, _ = self.attention(query, key, value, mask, dropout)
        # dimensions are bs, num_time_emb, num_ref_points, num_head * input_dim
        # transpose applied to flatten last 2 dimensions
        x = x.transpose(2, 3).contiguous().view(batch, -1, num_ref_points, self.h * dim)

        # out linear layer followed by summation over num_time_emb dimension
        return self.linears[-1](x).sum(1)
