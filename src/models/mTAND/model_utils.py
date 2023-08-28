import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def sample_z(mean, logstd, k_iwae):
    epsilon = torch.randn(k_iwae, mean.shape[0], mean.shape[1], mean.shape[2])
    z = epsilon * torch.exp(logstd) + mean  # modified
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
        mean_2 = torch.zeros_like(mean_1)
    if log_std_2 is None:
        log_std_2 = torch.zeros_like(log_std_1)
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
    sigma = torch.exp(log_std.float())

    out = (((x - mean) / sigma) ** 2) / 2
    out += log_std
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
