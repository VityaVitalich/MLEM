import torch
import torch.nn as nn
from . import preprocessors as prp
from ..trainers.losses import get_loss
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from .model_utils import out_to_padded_batch


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
        out = self.encoder(x_resized).view(
            bs, seq_len, self.num_features * self.feature_dim
        )
        # out.requires_grad_(True)
        return out


class BaseMixin(nn.Module):
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

        self.all_numeric_size = (
            len(self.data_conf.features.numeric_values)
            * self.model_conf.numeric_emb_size
        )

        self.input_dim = (
            all_emb_size + self.all_numeric_size + self.model_conf.use_deltas
        )

        ### NORMS ###
        self.pre_encoder_norm = getattr(nn, self.model_conf.pre_encoder_norm)(
            self.input_dim
        )
        self.post_encoder_norm = getattr(nn, self.model_conf.post_encoder_norm)(
            self.model_conf.encoder_hidden
        )
        self.decoder_norm = getattr(nn, self.model_conf.decoder_norm)(
            self.model_conf.decoder_hidden
        )
        self.encoder_norm = getattr(nn, self.model_conf.encoder_norm)(
            self.model_conf.encoder_hidden
        )

        ### MIXER ###
        if self.model_conf.encoder_feature_mixer:
            assert self.model_conf.features_emb_dim == self.model_conf.numeric_emb_size
            self.encoder_feature_mixer = FeatureMixer(
                num_features=len(self.data_conf.features.numeric_values)
                + len(self.data_conf.features.embeddings),
                feature_dim=self.model_conf.features_emb_dim,
                num_layers=1,
            )
        else:
            self.encoder_feature_mixer = nn.Identity()

        if self.model_conf.decoder_feature_mixer:
            assert self.model_conf.features_emb_dim == self.model_conf.numeric_emb_size
            self.decoder_feature_mixer = FeatureMixer(
                num_features=len(self.data_conf.features.numeric_values)
                + len(self.data_conf.features.embeddings),
                feature_dim=self.model_conf.features_emb_dim,
                num_layers=1,
            )
        else:
            self.decoder_feature_mixer = nn.Identity()

        ### INTERBATCH TRANSFORMER ###
        if self.model_conf.preENC_TR:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.input_dim,
                nhead=1,
                batch_first=self.model_conf.batch_first_encoder,
            )

            self.preENC_TR = nn.TransformerEncoder(
                encoder_layer,
                1,
                enable_nested_tensor=True,
                mask_check=True,
            )
        else:
            self.preENC_TR = nn.Identity()

        ### ENCODER ###
        if self.model_conf.encoder == "GRU":
            self.encoder = nn.GRU(
                self.input_dim,
                self.model_conf.encoder_hidden,
                batch_first=True,
                num_layers=self.model_conf.encoder_num_layers,
            )
        elif self.model_conf.encoder == "LSTM":
            self.encoder = nn.LSTM(
                self.input_dim,
                self.model_conf.encoder_hidden,
                batch_first=True,
                num_layers=self.model_conf.encoder_num_layers,
            )
        elif self.model_conf.encoder == "TR":
            self.encoder_proj = nn.Linear(
                self.input_dim, self.model_conf.encoder_hidden
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.model_conf.encoder_hidden,
                nhead=self.model_conf.encoder_num_heads,
                batch_first=True,
            )

            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                self.model_conf.encoder_num_layers,
                norm=self.encoder_norm,
                enable_nested_tensor=True,
                mask_check=True,
            )

        ### DECODER ###
        if self.model_conf.decoder == "GRU":
            self.decoder = DecoderGRU(
                input_size=self.input_dim,
                hidden_size=self.model_conf.decoder_hidden,
                global_hidden_size=self.model_conf.encoder_hidden,
                num_layers=self.model_conf.decoder_num_layers,
            )
        elif self.model_conf.decoder == "TR":
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.model_conf.decoder_hidden,
                nhead=self.model_conf.decoder_heads,
                batch_first=True,
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=self.model_conf.decoder_num_layers,
                norm=self.decoder_norm,
            )
            self.decoder_proj = nn.Linear(
                self.input_dim, self.model_conf.decoder_hidden
            )

        ### DROPOUT ###
        self.global_hid_dropout = nn.Dropout(self.model_conf.after_enc_dropout)

        ### ACTIVATION ###
        self.act = nn.GELU()

        ### OUT PROJECTION ###
        self.out_proj = nn.Linear(self.model_conf.decoder_hidden, self.input_dim)

        ### LOSS ###
        self.embedding_predictor = EmbeddingPredictor(
            model_conf=self.model_conf, data_conf=self.data_conf
        )
        self.numeric_projector = NumericalFeatureProjector(
            model_conf=self.model_conf, data_conf=self.data_conf
        )
        self.mse_fn = torch.nn.MSELoss(reduction="none")
        self.ce_fn = torch.nn.CrossEntropyLoss(
            reduction="mean", ignore_index=0, label_smoothing=0.15
        )

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

        ### SPARCE EMBEDDINGS ###
        sparce_loss = torch.mean(torch.sum(torch.abs(output["latent"]), dim=1))

        ### GENERATED EMBEDDINGS DISTANCE ###
        gen_embeddings_loss = self.generative_embedding_loss(output)

        losses_dict = {
            "total_mse_loss": total_mse_loss,
            "total_CE_loss": total_ce_loss,
            "delta_loss": self.model_conf.delta_weight * delta_mse_loss,
            "sparcity_loss": sparce_loss,
            "gen_embedding_loss": gen_embeddings_loss,
        }
        losses_dict.update(cross_entropy_losses)

        total_loss = (
            self.model_conf.mse_weight * losses_dict["total_mse_loss"]
            + self.model_conf.CE_weight * total_ce_loss
            + self.model_conf.delta_weight * delta_mse_loss
            + self.model_conf.l1_weight * sparce_loss
            + self.model_conf.gen_emb_weight * gen_embeddings_loss
        )
        losses_dict["total_loss"] = total_loss

        return losses_dict

    def numerical_loss(self, output):
        # MSE
        total_mse_loss = 0
        for key, values in output["gt"]["input_batch"].payload.items():
            if key in self.processor.numeric_names:
                gt_val = values.float()[:, 1:]
                pred_val = output["pred"][key].squeeze(-1)

                mse_loss = self.mse_fn(
                    gt_val,
                    pred_val,
                )
                mask = gt_val != 0
                masked_mse = mse_loss * mask
                total_mse_loss += (
                    masked_mse.sum(dim=1) / (mask != 0).sum(dim=1)
                ).mean()

        return total_mse_loss

    def delta_mse_loss(self, output):
        # DELTA MSE
        if self.model_conf.use_deltas:
            gt_delta = output["gt"]["time_steps"].diff(1)
            delta_mse = self.mse_fn(gt_delta, output["pred"]["delta"])
            mask = output["gt"]["time_steps"] != -1
            delta_masked = delta_mse * mask[:, 1:]
            delta_mse = delta_masked.sum() / (mask[:, 1:] != 0).sum()
        else:
            delta_mse = torch.tensor(0)

        return delta_mse

    def generative_embedding_loss(self, output):
        if self.model_conf.generative_embeddings_loss:
            gt = output["all_latents"][:, 1:, :].detach()
            gen = output["gen_all_latents"]
            # у кого рубить градиент?))))
            if self.model_conf.gen_emb_loss_type == "l2":
                loss = F.mse_loss(gen, gt, reduction="none").sum(dim=-1)
            elif self.model_conf.gen_emb_loss_type == "l1":
                loss = F.l1_loss(gen, gt, reduction="none").sum(dim=-1)
            elif self.model_conf.gen_emb_loss_type == "cosine":
                bs, seq_len, d = gen.size()
                loss = F.cosine_embedding_loss(
                    gen.reshape(bs * seq_len, d),
                    gt.reshape(bs * seq_len, d),
                    torch.ones(bs * seq_len, device=gt.device),
                    reduction="none",
                ).view(bs, seq_len)

            mask = output["gt"]["time_steps"] != -1
            #  print('gen loss', loss.size(), mask.size())
            loss = loss * mask[:, 1:]
            loss = loss.sum() / (mask[:, 1:] != 0).sum()
        else:
            loss = torch.tensor(0)

        return loss


class SeqGen(BaseMixin):
    def __init__(self, model_conf, data_conf):
        super().__init__(model_conf=model_conf, data_conf=data_conf)

    def forward(self, padded_batch, generative=False):
        x, time_steps = self.processor(padded_batch)
        x = self.encoder_feature_mixer(x)
        if self.model_conf.use_deltas:
            gt_delta = time_steps.diff(1)
            delta_feature = torch.cat(
                [gt_delta, torch.zeros(x.size()[0], 1, device=gt_delta.device)], dim=1
            )
            x = torch.cat([x, delta_feature.unsqueeze(-1)], dim=-1)

        x = self.preENC_TR(x)

        if self.model_conf.encoder in ("GRU", "LSTM"):
            all_hid, hn = self.encoder(self.pre_encoder_norm(x))
            lens = padded_batch.seq_lens - 1
            last_hidden = self.post_encoder_norm(all_hid[:, lens, :].diagonal().T)
        elif self.model_conf.encoder == "TR":
            x_proj = self.encoder_proj(x)
            all_hid = self.encoder(x_proj)
            last_hidden = all_hid[:, 0, :]

        last_hidden = self.global_hid_dropout(last_hidden)

        if self.model_conf.decoder == "GRU":
            dec_out = self.decoder(x, last_hidden)

        elif self.model_conf.decoder == "TR":
            x_proj = self.decoder_proj(x)
            mask = torch.nn.Transformer.generate_square_subsequent_mask(
                x.size(1), device=x.device
            )
            dec_out = self.decoder(
                tgt=x_proj,
                memory=last_hidden.unsqueeze(1),
                tgt_mask=mask,
            )

        out = self.out_proj(dec_out)
        if self.model_conf.use_deltas:
            out_mixed = self.decoder_feature_mixer(out[:, :, :-1])
            out = torch.cat([out_mixed, out[:, :, -1].unsqueeze(-1)], dim=-1)
        else:
            out = self.decoder_feature_mixer(out)
        out = out[:, :-1, :]

        pred = self.embedding_predictor(out)
        pred.update(self.numeric_projector(out))
        if self.model_conf.use_deltas:
            pred["delta"] = out[:, :, -1].squeeze(-1)

        gt = {"input_batch": padded_batch, "time_steps": time_steps}

        res_dict = {
            "gt": gt,
            "pred": pred,
            "latent": last_hidden,
        }

        if self.model_conf.generative_embeddings_loss:
            res_dict["all_latents"] = all_hid
            if not generative:
                gen_batch = out_to_padded_batch(res_dict, self.data_conf)
                generated_out = self.forward(
                    gen_batch.to(self.model_conf.device), generative=True
                )
                res_dict["gen_all_latents"] = generated_out["all_latents"]
                res_dict["gen_latent"] = generated_out["latent"]

        return res_dict


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
            if name in self.emb_names:
                shifted_labels = padded_batch.payload[name].long()[:, 1:]
                embed_losses[name] = self.criterion(
                    dist.permute(0, 2, 1), shifted_labels
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


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, global_hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.mix_global = nn.Linear(hidden_size + global_hidden_size, hidden_size)
        self.act = nn.GELU()
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

        hx = self.act(self.mix_global(torch.cat([global_hidden, hx], dim=-1)))
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

    def forward(self, input, global_hidden, hx=None, **kwargs):
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
