import torch
import torch.nn as nn
from einops import repeat, rearrange, reduce


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

        mask = (
            torch.arange(T, device=seq_lens.device).view(1, -1).repeat(B, 1)
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

        self.numeric_processor = nn.ModuleDict()
        self.numeric_norms = nn.ModuleDict()
        for name in self.numeric_names:
            if self.model_conf.use_numeric_emb:
                self.numeric_processor[name] = nn.Linear(
                    1, self.model_conf.numeric_emb_size
                )
            else:
                self.numeric_processor[name] = nn.Identity()

            self.numeric_norms[name] = RBatchNormWithLens()

    def forward(self, padded_batch, use_norm=True):
        numeric_values = []
        categoric_values = []

        time_steps = padded_batch.payload.get("event_time").float()
        seq_lens = padded_batch.seq_lens
        for key, values in padded_batch.payload.items():
            if key in self.emb_names:
                categoric_values.append(self.embed_layers[key](values.long()))
            elif key in self.numeric_names:
                if use_norm:
                    cur_value = self.numeric_norms[key](values.float(), seq_lens)
                else:  # we do not want to use normalization when applying decoder to our sequence
                    # otherwise it would be hard to make true generation and lead to bias in forecasting
                    cur_value = values.float().unsqueeze(-1)

                numeric_values.append(self.numeric_processor[key](cur_value))

        if len(categoric_values) == 0:
            return torch.cat(numeric_values, dim=-1), time_steps
        if len(numeric_values) == 0:
            return torch.cat(categoric_values, dim=-1), time_steps

        categoric_tensor = torch.cat(categoric_values, dim=-1)
        numeric_tensor = torch.cat(numeric_values, dim=-1)

        return torch.cat([categoric_tensor, numeric_tensor], dim=-1), time_steps


class TimeConcater(nn.Module):
    def __init__(self, num_time_blocks, device):
        super().__init__()
        blocks = torch.linspace(0.0, 1.001, num_time_blocks + 1)
        lr = []
        for i in range(len(blocks) - 1):
            lr.append(torch.concat([blocks[i].view(1, 1), blocks[i + 1].view(1, 1)]))

        self.ls, self.rs = torch.concat(lr, dim=1).to(device)
        self.num_points = num_time_blocks

    def _create_idx_mask(self, time_steps):
        bs, seq_len = time_steps.size()

        left_idx = repeat(time_steps, "b l -> p b l", p=self.num_points) >= repeat(
            self.ls, "p -> p b l", b=bs, l=seq_len
        )
        right_idx = repeat(time_steps, "b l -> p b l", p=self.num_points) < repeat(
            self.rs, "p -> p b l", b=bs, l=seq_len
        )

        multi_idx = left_idx & right_idx

        return multi_idx

    def forward(self, x, time_steps):
        """
        x - tensor of size (BS, L, D)
        time steps - tensor of size (BS, L)

        returns x of size (BS, Num ref points, in dim) and time steps
        """
        in_dim = x.size()[-1]
        mask = self._create_idx_mask(time_steps)
        new_x = reduce(
            (
                repeat(x, "b l d -> p b l d", p=self.num_points)
                * repeat(mask, "p b l -> p b l d", d=in_dim)
            ),
            "p b l d -> b p d",
            "sum",
        )
        return new_x, time_steps


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, time_steps):
        """
        x - tensor of size (BS, L, D)
        time steps - tensor of size (BS, L)

        returns the same
        """
        return x, time_steps


class MultiTimeSummator(nn.Module):
    def __init__(self, time_blocks, device):
        super().__init__()

        assert isinstance(time_blocks, list)

        self.time_ps = []
        for num_block in time_blocks:
            self.time_ps.append(TimeConcater(num_block, device))

        self.max_len = max(time_blocks)

        self.weights = nn.Parameter(torch.ones(len(time_blocks)) / len(time_blocks))
        # self.weights.data[-1] = 1
        # self.weights.data[:-1] = 1e-10
        # print(self.weights.data)
        # self.dropout = nn.Dropout1d(p=0.0)

    def forward(self, x, time_steps):
        new_xs = self.collect_new_x(x, time_steps)

        nb, bs, l, d = new_xs.size()

        self.softmaxed_weights = nn.functional.softmax(self.weights, dim=0)
        # if self.training:
        #     self.softmaxed_weights = nn.functional.gumbel_softmax(self.weights, tau=2, hard=True)
        # else:
        #     self.softmaxed_weights = torch.zeros_like(self.weights)
        #     self.softmaxed_weights[self.weights.argmax()] = 1
        # self.softmaxed_weights = self.weights

        cur_w = repeat(self.softmaxed_weights, "nb -> nb bs l d", bs=bs, l=l, d=d)

        out = (new_xs * cur_w).sum(dim=0)

        return out, time_steps

    def collect_new_x(self, x, time_steps):
        new_xs = []

        for i, tc in enumerate(self.time_ps):
            cur_multiplier = self.max_len // tc.num_points
            out_x, time_steps = tc(x, time_steps)
            new_x = self.dropout(
                torch.repeat_interleave(out_x, cur_multiplier, dim=1)
            ).unsqueeze(0)

            # if i < (len(self.time_ps) - 1):
            #     new_x = new_x.detach()

            new_xs.append(new_x)

        return torch.cat(new_xs, dim=0)


class TimeEncoder(nn.Module):
    def __init__(self, model_conf, data_conf):
        super().__init__()
        self.model_conf = model_conf
        self.data_conf = data_conf

        if self.model_conf.time_embedding and self.model_conf.use_deltas:
            self.alpha = nn.Parameter(torch.rand(self.model_conf.time_embedding))

    def forward(self, x, time_steps):
        if not self.model_conf.use_deltas:
            return x

        gt_delta = time_steps.diff(1)
        delta_feature = torch.cat(
            [torch.zeros(x.size(0), 1, device=gt_delta.device), gt_delta], dim=1
        )
        time_emb = self.create_emb(delta_feature)

        x = torch.cat([x, time_emb], dim=-1)
        return x

    def create_emb(self, delta_feature):
        if self.model_conf.time_embedding:
            bs, l = delta_feature.size()
            return torch.cat(
                [
                    torch.sin(delta_feature)
                    .view(bs, l, 1)
                    .repeat(1, 1, self.model_conf.time_embedding)
                    * self.alpha,
                    torch.cos(delta_feature)
                    .view(bs, l, 1)
                    .repeat(1, 1, self.model_conf.time_embedding)
                    * self.alpha,
                ],
                dim=-1,
            )
        else:
            return delta_feature.unsqueeze(-1)
