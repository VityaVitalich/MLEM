import torch
import torch.nn as nn
from . import preprocessors as prp
from ..trainers.losses import get_loss
import numpy as np
from torch.autograd import Variable


class L2Normalization(nn.Module):
    def __init__(self):
        super(L2Normalization, self).__init__()

    def forward(self, x):
        return x.div(torch.norm(x, dim=1).view(-1, 1))


class BaseMixin(nn.Module):
    def __init__(self, model_conf, data_conf):
        super().__init__()
        self.model_conf = model_conf
        self.data_conf = data_conf

        ### PROCESSORS ###
        self.processor = prp.FeatureProcessor(
            model_conf=model_conf, data_conf=data_conf
        )
        self.time_processor = getattr(prp, self.model_conf.time_preproc)(
            self.model_conf.num_time_blocks, model_conf.device
        )

        ### INPUT SIZE ###
        all_emb_size = self.model_conf.features_emb_dim * len(
            self.data_conf.features.embeddings
        )
        all_numeric_size = len(self.data_conf.features.numeric_values)
        self.input_dim = all_emb_size + all_numeric_size

        ### NORMS ###
        self.pre_gru_norm = getattr(nn, self.model_conf.pre_gru_norm)(self.input_dim)
        self.post_gru_norm = getattr(nn, self.model_conf.post_gru_norm)(
            self.model_conf.classifier_gru_hidden_dim
        )
        self.encoder_norm = getattr(nn, self.model_conf.encoder_norm)(self.input_dim)

        ### TRANSFORMER ###
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_dim, nhead=self.model_conf.num_heads_enc
        )

        self.encoder = getattr(nn, self.model_conf.encoder)(
            encoder_layer,
            self.model_conf.num_enc_layers,
            norm=self.encoder_norm,
            enable_nested_tensor=True,
            mask_check=True,
        )

        ### DROPOUT ###
        self.after_enc_dropout = nn.Dropout1d(self.model_conf.after_enc_dropout)

        ### OUT PROJECTION ###
        self.out_linear = getattr(nn, self.model_conf.predict_head)(
            self.model_conf.classifier_gru_hidden_dim, self.data_conf.num_classes
        )

        ### LOSS ###

        self.loss_fn = get_loss(self.model_conf)

    def loss(self, out, gt):
        if self.model_conf.predict_head == "Identity":
            loss = self.loss_fn(out, gt[0])
        else:
            loss = self.loss_fn(out, gt[1])

        if self.model_conf.time_preproc == "MultiTimeSummator":
            # logp = torch.log(self.time_processor.softmaxed_weights)
            # entropy_term = torch.sum(-self.time_processor.softmaxed_weights * logp)
            entropy_term = torch.tensor(0)
        else:
            entropy_term = torch.tensor(0)
        return {
            "total_loss": loss + self.model_conf.entropy_weight * entropy_term,
            "entropy_loss": entropy_term,
        }


class GRUGen(nn.Module):
    def __init__(self, model_conf, data_conf):
        super().__init__()

        self.model_conf = model_conf
        self.data_conf = data_conf

        ### PROCESSORS ###
        self.processor = prp.FeatureProcessor(
            model_conf=model_conf, data_conf=data_conf
        )

        ### INPUT SIZE ###
        all_emb_size = self.model_conf.features_emb_dim * len(
            self.data_conf.features.embeddings
        )
        self.all_numeric_size = len(self.data_conf.features.numeric_values)
        self.input_dim = all_emb_size + self.all_numeric_size

        self.encoder = nn.GRU(
            self.input_dim,
            self.model_conf.encoder_hidden,
            batch_first=True,
        )
        self.decoder = DecoderGRU(
            input_size=self.input_dim,
            hidden_size=self.model_conf.decoder_gru_hidden,
            global_hidden_size=self.model_conf.encoder_hidden,
            num_layers=self.model_conf.decoder_num_layers,
        )
        self.out_proj = nn.Linear(self.model_conf.decoder_gru_hidden, self.input_dim)
        self.delta_proj = nn.Linear(self.model_conf.decoder_gru_hidden, 1)

        self.embedding_predictor = EmbeddingPredictor(
            model_conf=self.model_conf, data_conf=self.data_conf
        )
        self.mse_fn = torch.nn.MSELoss(reduction="none")
        self.ce_fn = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=0)

    def forward(self, padded_batch):
        x, time_steps = self.processor(padded_batch)
        all_hid, hn = self.encoder(x)

        lens = padded_batch.seq_lens - 1
        last_hidden = all_hid[:, lens, :].diagonal().T

        dec_out = self.decoder(x, last_hidden)
        out = self.out_proj(dec_out)[:, :-1, :]
        pred_delta = self.delta_proj(dec_out)[:, :-1, :].squeeze(-1)

        emb_dist = self.embedding_predictor(out)

        return {
            "x": x,
            "time_steps": time_steps,
            "pred": out,
            "pred_delta": pred_delta,
            "input_batch": padded_batch,
            "emb_dist": emb_dist,
            "latent": last_hidden,
        }

    def loss(self, output, ground_truth):
        """
        output: Dict that is outputed from forward method
        """
        gt_embedding = output["x"][:, 1:, :]
        pred = output["pred"]

        # MSE
        mse_loss = self.mse_fn(
            gt_embedding[:, :, self.all_numeric_size :],
            pred[:, :, self.all_numeric_size :],
        )
        mask = gt_embedding[:, :, self.all_numeric_size :] != 0
        masked_mse = mse_loss * mask
        mse_loss = masked_mse.sum() / (masked_mse != 0).sum()

        # DELTA MSE
        gt_delta = output["time_steps"].diff(1)
        delta_mse = self.mse_fn(gt_delta, output["pred_delta"])
        mask = output["time_steps"] != -1
        delta_masked = delta_mse * mask[:, 1:]
        delta_mse = delta_masked.sum() / (delta_masked != 0).sum()

        # CROSS ENTROPY
        cross_entropy_losses = self.embedding_predictor.loss(
            output["emb_dist"], output["input_batch"]
        )
        total_ce_loss = torch.sum(
            torch.cat([value.unsqueeze(0) for _, value in cross_entropy_losses.items()])
        )

        losses_dict = {
            "mse_loss": mse_loss,
            "total_CE_loss": total_ce_loss,
            "delta_loss": self.model_conf.delta_weight * delta_mse,
        }
        losses_dict.update(cross_entropy_losses)

        total_loss = (
            self.model_conf.mse_weight * losses_dict["mse_loss"]
            + self.model_conf.CE_weight * total_ce_loss
            + self.model_conf.delta_weight * delta_mse
        )
        losses_dict["total_loss"] = total_loss

        return losses_dict


class EmbeddingPredictor(nn.Module):
    def __init__(self, model_conf, data_conf):
        super().__init__()
        self.model_conf = model_conf
        self.data_conf = data_conf

        self.criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=0)

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
            shifted_labels = padded_batch.payload[name].long()[:, 1:]
            embed_losses[name] = self.criterion(dist.permute(0, 2, 1), shifted_labels)

        return embed_losses


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, global_hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.mix_global = nn.Linear(hidden_size + global_hidden_size, hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, global_hidden, hx=None):
        # Inputs:
        #       input: of shape (batch_size, input_size)
        #       global_hidden: of shape (batch_size, global_hidden_size)
        #       hx: of shape (batch_size, hidden_size)
        # Output:
        #       hy: of shape (batch_size, hidden_size)

        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), self.hidden_size))

        hx = self.mix_global(torch.cat([global_hidden, hx], dim=-1))
        x_t = self.x2h(input)
        h_t = self.h2h(hx)

        x_reset, x_upd, x_new = x_t.chunk(3, 1)
        h_reset, h_upd, h_new = h_t.chunk(3, 1)

        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_upd + h_upd)
        new_gate = torch.tanh(x_new + (reset_gate * h_new))

        hy = update_gate * hx + (1 - update_gate) * new_gate

        return hy


class DecoderGRU(nn.Module):
    def __init__(
        self, input_size, hidden_size, global_hidden_size, num_layers, bias=True
    ):
        super(DecoderGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.global_hidden_size = global_hidden_size
        self.num_layers = num_layers
        self.bias = bias

        self.rnn_cell_list = nn.ModuleList()
        self.rnn_cell_list.append(
            GRUCell(
                self.input_size, self.hidden_size, self.global_hidden_size, self.bias
            )
        )
        for l in range(1, self.num_layers):
            self.rnn_cell_list.append(
                GRUCell(
                    self.hidden_size,
                    self.hidden_size,
                    self.global_hidden_size,
                    self.bias,
                )
            )

    def forward(self, input, global_hidden, hx=None):
        # Input of shape (batch_size, seqence length, input_size)
        #
        # Output of shape (batch_size, output_size)

        if hx is None:
            h0 = Variable(
                torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(
                    input.device
                )
            )

        else:
            h0 = hx

        outs = []

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append(h0[layer, :, :])

        for t in range(input.size(1)):
            for layer in range(self.num_layers):
                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](
                        input[:, t, :], global_hidden, hidden[layer]
                    )
                else:
                    hidden_l = self.rnn_cell_list[layer](
                        hidden[layer - 1], global_hidden, hidden[layer]
                    )
                hidden[layer] = hidden_l

                hidden[layer] = hidden_l

            outs.append(hidden_l.unsqueeze(1))

        return torch.cat(outs, dim=1)
