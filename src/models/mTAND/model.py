import numpy as np
import torch
import torch.nn as nn

from .model_utils import multiTimeAttention


class enc_mtan_rnn(nn.Module):
    def __init__(
        self,
        input_dim,
        query,
        latent_dim=2,
        nhidden=16,
        embed_time=16,
        num_heads=1,
        linear_hidden_dim=50,
        learn_emb=False,
        device="cuda",
    ):
        super(enc_mtan_rnn, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.learn_emb = learn_emb
        self.linear_hidden_dim = linear_hidden_dim
        self.att = multiTimeAttention(input_dim, nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(nhidden, nhidden, bidirectional=True, batch_first=True)
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(2 * nhidden, self.linear_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.linear_hidden_dim, latent_dim * 2),
        )
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time - 1)
            self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def fixed_time_embedding(self, pos):
        d_model = self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.0 * pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x, time_steps):
        time_steps = time_steps.cpu()
        mask = x[:, :, self.dim :]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps).to(self.device)
            query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
        else:
            key = self.fixed_time_embedding(time_steps).to(self.device)
            query = self.fixed_time_embedding(self.query.unsqueeze(0)).to(self.device)
        out = self.att(query, key, x, mask=None)
        out, _ = self.gru_rnn(out)
        out = self.hiddens_to_z0(out)
        return out


class FeatureProcessor(nn.Module):
    def __init__(self, model_conf, data_conf):
        super(FeatureProcessor, self).__init__()
        self.model_conf = model_conf
        self.data_conf = data_conf

        self.emb_names = list(self.data_conf.features.embeddings.keys())
        self.init_embed_layers()

    def init_embed_layers(self):
        self.embed_layers = nn.ModuleDict()

        for name in self.emb_names:
            vocab_size = self.data_conf.features.embeddings[name]["max_value"]
            self.embed_layers[name] = nn.Embedding(
                vocab_size, self.model_conf.features_emb_dim
            )

    def forward(self, padded_batch):
        numeric_values = []

        for key, values in padded_batch.payload.items():
            if key in self.emb_names:
                numeric_values.append(self.embed_layers[key](values))
            else:
                if key == "event_time":
                    time_steps = values
                else:
                    numeric_values.append(values.unsqueeze(-1).float())

        x = torch.cat(numeric_values, dim=-1)
        return x, time_steps


class MegaEncoder(nn.Module):
    def __init__(self, model_conf, data_conf):
        super(MegaEncoder, self).__init__()
        self.model_conf = model_conf
        self.data_conf = data_conf

        all_emb_size = self.model_conf.features_emb_dim * len(
            self.data_conf.features.embeddings
        )
        all_numeric_size = len(self.data_conf.features.numeric_values)
        self.input_dim = all_emb_size + all_numeric_size

        self.preprocessor = FeatureProcessor(
            model_conf=self.model_conf, data_conf=self.data_conf
        )

        self.ref_points = torch.linspace(0.0, 1.0, self.model_conf.num_ref_points)
        self.encoder = enc_mtan_rnn(
            self.input_dim,
            self.ref_points,
            latent_dim=self.model_conf.latent_dim,
            nhidden=self.model_conf.ref_point_dim,
            embed_time=self.model_conf.time_emb_dim,
            num_heads=self.model_conf.num_heads_enc,
            linear_hidden_dim=self.model_conf.linear_hidden_dim,
            learn_emb=True,
            device=self.model_conf.device,
        )

    def forward(self, padded_batch):
        x, time_steps = self.preprocessor(padded_batch)
        out = self.encoder(x, time_steps.float())

        return out
