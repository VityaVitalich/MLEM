import torch
import torch.nn as nn


class RBatchNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.bn = torch.nn.BatchNorm1d(1)

    def forward(self, x):
        B, T, _ = x.size()  # B x T x 1
        x = x.view(B * T, 1)
        x = self.bn(x)
        x = x.view(B, T, 1)
        return x


class RBatchNormWithLens(torch.nn.Module):
    """
    The same as RBatchNorm, but ...
    Drop padded symbols (zeros) from batch when batch stat update
    """

    def __init__(self):
        super().__init__()

        self.bn = torch.nn.BatchNorm1d(1)

    def forward(self, x, seq_lens):
        B, T = x.size()  # B x T

        mask = torch.arange(T, device=seq_lens.device).view(1, -1).repeat(
            B, 1
        ) < seq_lens.view(-1, 1)
        x_new = x
        x_new[mask] = self.bn(x[mask].view(-1, 1)).view(-1)
        return x_new.view(B, T, 1)


class FeatureProcessor(nn.Module):
    def __init__(self, model_conf, data_conf):
        super().__init__()
        self.model_conf = model_conf
        self.data_conf = data_conf

        self.emb_names = list(self.data_conf.features.embeddings.keys())
        self.numeric_names = list(self.data_conf.features.numeric_values.keys())
        self.init_embed_layers()

    def init_embed_layers(self):
        self.embed_layers = nn.ModuleDict()

        for name in self.emb_names:
            vocab_size = self.data_conf.features.embeddings[name]["max_value"]
            self.embed_layers[name] = nn.Embedding(
                vocab_size, self.model_conf.features_emb_dim
            )

        self.numeric_norms = nn.ModuleDict()
        for name in self.numeric_names:
            self.numeric_norms[name] = RBatchNormWithLens()

    def forward(self, padded_batch):
        numeric_values = []
        categoric_values = []

        time_steps = padded_batch.payload.pop("event_time").float()
        seq_lens = padded_batch.seq_lens
        for key, values in padded_batch.payload.items():
            if key in self.emb_names:
                categoric_values.append(self.embed_layers[key](values.long()))
            else:
                # TODO: repeat the numerical feature?
                numeric_values.append(self.numeric_norms[key](values.float(), seq_lens))

        if len(categoric_values) == 0:
            return torch.cat(numeric_values, dim=-1), time_steps

        categoric_tensor = torch.cat(categoric_values, dim=-1)
        numeric_tensor = torch.cat(numeric_values, dim=-1)

        return torch.cat([categoric_tensor, numeric_tensor], dim=-1), time_steps
