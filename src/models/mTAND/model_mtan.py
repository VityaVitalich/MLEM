import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from ..preprocessors import FeatureProcessor


class MultiTimeAttention(nn.Module):
    def __init__(self, input_dim, nhidden=16, embed_time=16, num_heads=1):
        super(MultiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
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
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim=-2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn * value.unsqueeze(-3), -2), p_attn

    def forward(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [
            l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key))
        ]
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * dim)
        return self.linears[-1](x)


class EncMtanRnnClassification(nn.Module):
    def __init__(
        self,
        input_dim,
        query,
        latent_dim=2,
        nhidden=16,
        embed_time=16,
        num_heads=1,
        linear_hidden_dim=50,
        num_time_emb=3,
        device="cuda",
    ):
        super().__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.linear_hidden_dim = linear_hidden_dim
        self.num_time_emb = num_time_emb
        self.att = MultiTimeAttention(input_dim, nhidden, embed_time, num_heads)

        self.classifier = nn.Sequential(
            nn.Linear(nhidden, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 2),
        )
        self.enc = nn.GRU(nhidden, nhidden)
        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def forward(self, x, time_steps):
        time_steps = time_steps.to(self.device)

        key = self.learn_time_embedding(time_steps).to(self.device)
        query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)

        out = self.att(query, key, x, mask=None)
        out = out.permute(1, 0, 2)
        _, out = self.enc(out)
        return self.classifier(out.squeeze(0))


class MegaNetClassifier(nn.Module):
    def __init__(self, data_conf, model_conf):
        super().__init__()
        self.model_conf = model_conf
        self.data_conf = data_conf

        self.ref_points = torch.linspace(0.0, 1.0, self.model_conf.num_ref_points).to(
            self.model_conf.device
        )

        self.preprocessor = FeatureProcessor(
            model_conf=self.model_conf, data_conf=self.data_conf
        )

        all_emb_size = self.model_conf.features_emb_dim * len(
            self.data_conf.features.embeddings
        )
        all_numeric_size = len(self.data_conf.features.numeric_values)
        self.input_dim = all_emb_size + all_numeric_size

        self.encoder = EncMtanRnnClassification(
            self.input_dim,
            self.ref_points,
            latent_dim=self.model_conf.latent_dim,
            nhidden=self.model_conf.ref_point_dim,
            embed_time=self.model_conf.time_emb_dim,
            num_heads=self.model_conf.num_heads_enc,
            linear_hidden_dim=self.model_conf.linear_hidden_dim,
            num_time_emb=self.model_conf.num_time_emb,
            device=self.model_conf.device,
        )

    def forward(self, padded_batch):
        x, time_steps = self.preprocessor(padded_batch)
        enc_out = self.encoder(x, time_steps)

        return {
            "x": x,
            "time_steps": time_steps,
            "y_pred": enc_out,
        }

    def loss(self, output, ground_truth):
        """
        output: Dict that is outputed from forward method
        should contain
        1) reconstructed x
        2) initial x
        3) mu from latent
        4) log_std from latent

        ground truth is a Tuple with idx at first pos and label at second
        """

        classification_loss = nn.functional.cross_entropy(
            output["y_pred"], ground_truth[1]
        )

        return {"total_loss": classification_loss}
