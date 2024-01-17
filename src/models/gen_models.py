import torch
import torch.nn as nn
from . import preprocessors as prp
from ..trainers.losses import get_loss
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from .model_utils import (
    out_to_padded_batch,
    FeatureMixer,
    L2Normalization,
    set_grad,
    EmbeddingPredictor,
    NumericalFeatureProjector,
)
from functools import partial
from einops import repeat


class BaseMixin(nn.Module):
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
        if self.model_conf.time_embedding:
            self.input_dim += self.model_conf.time_embedding * 2 - 1

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
            if self.model_conf.time_embedding:
                assert (
                    self.model_conf.features_emb_dim
                    == self.model_conf.time_embedding * 2
                )

                self.encoder_feature_mixer = FeatureMixer(
                    num_features=len(self.data_conf.features.numeric_values)
                    + len(self.data_conf.features.embeddings)
                    + 1,
                    feature_dim=self.model_conf.features_emb_dim,
                    num_layers=1,
                )
            else:
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
            self.encoder_proj = nn.Identity()
        elif self.model_conf.encoder == "LSTM":
            self.encoder = nn.LSTM(
                self.input_dim,
                self.model_conf.encoder_hidden,
                batch_first=True,
                num_layers=self.model_conf.encoder_num_layers,
            )
            self.encoder_proj = nn.Identity()
        elif self.model_conf.encoder == "TR":
            self.enc_pos_encoding = nn.Embedding(
                self.data_conf.test.max_seq_len + 1, self.model_conf.encoder_hidden
            )
            self.cls_token = nn.Parameter(torch.rand(self.model_conf.encoder_hidden))
            self.encoder_proj = nn.Linear(
                self.input_dim, self.model_conf.encoder_hidden
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.model_conf.encoder_hidden,
                # d_model=self.input_dim,
                nhead=self.model_conf.encoder_num_heads,
                batch_first=True,
                dim_feedforward=self.model_conf.encoder_dim_ff,
            )

            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                self.model_conf.encoder_num_layers,
                norm=self.encoder_norm,
                enable_nested_tensor=True,
                mask_check=True,
            )

        ### HIDDEN TO X0 PROJECTION ###
        self.hidden_to_x0 = nn.Linear(self.model_conf.encoder_hidden, self.input_dim)

        ### DECODER ###
        if self.model_conf.decoder == "GRU":
            self.decoder = DecoderGRU(
                input_size=self.input_dim,
                hidden_size=self.model_conf.decoder_hidden,
                global_hidden_size=self.model_conf.encoder_hidden,
                num_layers=self.model_conf.decoder_num_layers,
            )
        elif self.model_conf.decoder == "TR":
            self.dec_pos_encoding = nn.Embedding(
                self.data_conf.test.max_seq_len + 1, self.model_conf.decoder_hidden
            )
            self.decoder = TransformerDecoder(
                d_model=self.model_conf.decoder_hidden,
                nhead=self.model_conf.decoder_heads,
                num_layers=self.model_conf.decoder_num_layers,
                norm=self.decoder_norm,
                pos_encoding=self.dec_pos_encoding,
                dim_feedforward=self.model_conf.decoder_dim_ff,
            )
            self.decoder_proj = nn.Linear(
                self.input_dim, self.model_conf.decoder_hidden
            )

        ### OUT PROJECTION ###
        if self.model_conf.time_embedding:
            self.out_proj = nn.Linear(
                self.model_conf.decoder_hidden,
                self.input_dim - (self.model_conf.time_embedding * 2) + 1,
            )
            # substraction needed to predict only 1 value
        else:
            self.out_proj = nn.Linear(self.model_conf.decoder_hidden, self.input_dim)

        self.dec_out_proj = partial(self.decoder_out_projection_func)

        ### DROPOUT ###
        self.global_hid_dropout = nn.Dropout(self.model_conf.after_enc_dropout)

        ### ACTIVATION ###
        self.act = nn.GELU()

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

        self.register_encoder_layers()

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
        total_mse_loss = torch.tensor(0.0, device=self.model_conf.device)
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
                    masked_mse.sum(dim=1) # / (mask != 0).sum(dim=1)
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

    def generative_embedding_loss(self, output):
        if self.model_conf.generative_embeddings_loss:
            gt = output["all_latents"].detach()
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
            loss = loss * mask
            loss = loss.sum() / (mask != 0).sum()
        else:
            loss = torch.tensor(0)

        return loss

    def decoder_out_projection_func(
        self,
        dec_out,
        generation=False,
    ):
        # project to x + delta
        out = self.out_proj(dec_out)
        if len(dec_out.size()) == 2:  # used when generated in rnn
            out = out.unsqueeze(1)

        # if we use deltas then we need to mix features without them
        if self.model_conf.use_deltas:
            out_mixed = self.decoder_feature_mixer(out[:, :, :-1])

            delta = out[:, :, -1]
            if generation and self.model_conf.use_log_delta:
                delta = torch.exp(delta)
            # if we use time embedding, we expect input as time_embedding
            # but this should happen only during generation
            if generation:
                if self.model_conf.time_embedding:
                    prev_input = torch.cat(
                        [out_mixed, self.time_encoder.create_emb(delta)], dim=-1
                    )
                else:
                    prev_input = torch.cat([out_mixed, delta.unsqueeze(-1)], dim=-1)

            out = torch.cat([out_mixed, delta.unsqueeze(-1)], dim=-1)
        else:
            out = self.decoder_feature_mixer(out)

        # print(out.size())
        if len(dec_out.size()) == 2:  # used when generated in rnn
            out = out.squeeze(1)

        if generation:
            if self.model_conf.decoder == "TR":
                prev_input = self.decoder_proj(self.act(prev_input))

            if len(dec_out.size()) == 2:  # used when generated in rnn
                prev_input = prev_input.squeeze(1)
            return prev_input, out

        return out

    def register_encoder_layers(self):
        self.encoder_layers = [
            self.processor,
            self.pre_encoder_norm,
            self.encoder_norm,
            self.post_encoder_norm,
            self.encoder_feature_mixer,
            self.preENC_TR,
            self.encoder_proj,
            self.encoder,
        ]


class SeqGen(BaseMixin):
    def __init__(self, model_conf, data_conf):
        super().__init__(model_conf=model_conf, data_conf=data_conf)

    def forward(self, padded_batch):
        all_hidden, global_hidden, time_steps = self.encode(padded_batch)
        pred = self.decode(padded_batch, global_hidden)

        gt = {"input_batch": padded_batch, "time_steps": time_steps}

        res_dict = {
            "gt": gt,
            "pred": pred,
            "latent": global_hidden,
        }

        if self.model_conf.generative_embeddings_loss:
            res_dict["all_latents"] = all_hidden
            gen_batch = out_to_padded_batch(res_dict, self.data_conf)
            set_grad(self.encoder_layers, False)
            gen_all_hidden, gen_global_hidden, time_steps = self.encode(
                gen_batch.to(self.model_conf.device)
            )
            set_grad(self.encoder_layers, True)
            res_dict["gen_all_latents"] = gen_all_hidden
            res_dict["gen_latent"] = gen_global_hidden

        return res_dict

    def encode(self, padded_batch):
        x, time_steps = self.processor(padded_batch)

        if self.model_conf.time_embedding:
            x = self.time_encoder(x, time_steps)
            x = self.encoder_feature_mixer(x)
        else:
            x = self.encoder_feature_mixer(x)
            x = self.time_encoder(x, time_steps)

        x = self.preENC_TR(x)

        if self.model_conf.encoder in ("GRU", "LSTM"):
            all_hid, hn = self.encoder(self.pre_encoder_norm(x))
            lens = padded_batch.seq_lens - 1
            global_hidden = self.post_encoder_norm(all_hid[:, lens, :].diagonal().T)
        elif self.model_conf.encoder == "TR":
            x_proj = self.encoder_proj(x)
            # x_proj = x
            x_proj = torch.cat(
                [repeat(self.cls_token, "d -> b l d", b=x_proj.size(0), l=1), x_proj],
                dim=1,
            )
            x_proj = x_proj + self.enc_pos_encoding(
                torch.arange(x_proj.size(1), device=self.model_conf.device)
            )
            all_hid = self.encoder(x_proj)
            global_hidden = all_hid[:, 0, :]  # self.encoder_proj(all_hid[:, 0, :])

        global_hidden = self.global_hid_dropout(global_hidden)

        return all_hid, global_hidden, time_steps

    def decode(self, padded_batch, global_hidden):
        x, time_steps = self.processor(padded_batch, use_norm=False)
        x = self.time_encoder(x, time_steps)

        x0 = self.hidden_to_x0(global_hidden)
        x = torch.cat([x0.unsqueeze(1), x], dim=1)
        if self.model_conf.decoder == "GRU":
            dec_out = self.decoder(x, global_hidden)

        elif self.model_conf.decoder == "TR":
            x_proj = self.decoder_proj(self.act(x))
            mask = torch.nn.Transformer.generate_square_subsequent_mask(
                x.size(1), device=x.device
            )
            x_proj = x_proj + self.dec_pos_encoding(
                torch.arange(x_proj.size(1), device=self.model_conf.device)
            )
            # print(x_proj.size(), global_hidden.size())
            dec_out = self.decoder(
                tgt=x_proj,
                memory=x_proj[:, 0, :].unsqueeze(
                    1
                ),  # we can not pass global hidden due to dimension mismatch
                tgt_mask=mask,
            )

        out = self.dec_out_proj(dec_out)
        out = out[:, :-1, :]

        pred = self.embedding_predictor(out)
        pred.update(self.numeric_projector(out))
        if self.model_conf.use_deltas:
            pred["delta"] = torch.abs(out[:, :, -1].squeeze(-1))

        return pred

    def generate_sequence(self, global_hidden, lens):
        x0 = self.hidden_to_x0(global_hidden)
        if self.model_conf.decoder == "TR":
            global_hidden = self.decoder_proj(self.act(x0))
            x0 = global_hidden

        gens = self.decoder.generate(
            global_hidden=global_hidden,
            length=lens,
            pred_layers=self.dec_out_proj,
            first_step=x0,
        )
        pred = self.embedding_predictor(gens)
        pred.update(self.numeric_projector(gens))
        if self.model_conf.use_deltas:
            pred["delta"] = torch.abs(gens[:, :, -1].squeeze(-1))

        return pred

    def generate(self, padded_batch, lens):
        all_hid, global_hidden, time_steps = self.encode(padded_batch)
        return {"pred": self.generate_sequence(global_hidden, lens)}


class GenContrastive(SeqGen):
    def __init__(self, model_conf, data_conf):
        super().__init__(model_conf=model_conf, data_conf=data_conf)
        self.contrastive_loss_fn = get_loss(self.model_conf)

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

        ### CONTRASTIVE LOSS ###
        contrastive_loss = (
            self.model_conf.loss.contrastive_weight
            * self.contrastive_loss_fn(
                output["latent"], ground_truth[0].to(output["latent"].device)
            )
        )
        losses_dict["contrastive_loss"] = contrastive_loss

        total_loss = (
            self.model_conf.mse_weight * losses_dict["total_mse_loss"]
            + self.model_conf.CE_weight * total_ce_loss
            + self.model_conf.delta_weight * delta_mse_loss
            + self.model_conf.l1_weight * sparce_loss
            + self.model_conf.gen_emb_weight * gen_embeddings_loss
        )
        losses_dict["total_loss"] = (
            self.model_conf.loss.reconstruction_weight * total_loss + contrastive_loss
        )

        return losses_dict


class GenSigmoid(SeqGen):
    def __init__(self, model_conf, data_conf):
        super().__init__(model_conf=model_conf, data_conf=data_conf)

        init_temp = torch.tensor(10.0)
        init_bias = torch.tensor(-10.0)
        self.sigmoid_temp = nn.Parameter(torch.log(init_temp))
        self.sigmoid_bias = nn.Parameter(init_bias)


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

        hx = hx
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

            outs.append(hidden_l.unsqueeze(1))

        return torch.cat(outs, dim=1)

    def generate(self, global_hidden, length, pred_layers, first_step=None):
        h0 = Variable(
            torch.zeros(self.num_layers, global_hidden.size(0), self.hidden_size).to(
                global_hidden.device
            )
        )

        if first_step is None:
            first_step = torch.zeros(
                global_hidden.size(0), self.input_size, device=global_hidden.device
            )

        outs = []

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append(h0[layer, :, :])

        prev_input = first_step
        for t in range(length):
            for layer in range(self.num_layers):
                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](
                        prev_input, global_hidden, hidden[layer]
                    )
                else:
                    hidden_l = self.rnn_cell_list[layer](
                        hidden[layer - 1], global_hidden, hidden[layer]
                    )

                hidden[layer] = hidden_l

            prev_input, cur_output = pred_layers(hidden_l, generation=True)
            outs.append(cur_output.unsqueeze(1))

        return torch.cat(outs, dim=1)


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, norm, pos_encoding, dim_feedforward):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=dim_feedforward,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=norm,
        )

        self.pos_encoding = pos_encoding

    def forward(self, tgt, memory, tgt_mask):
        return self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)

    def generate(self, global_hidden, length, pred_layers, first_step):
        in_seq = torch.zeros(
            global_hidden.size(0),
            length + 1,
            global_hidden.size(1),
            device=global_hidden.device,
        )
        in_seq[:, 0, :] = first_step + self.pos_encoding(
            torch.tensor(0, device=global_hidden.device)
        )
        outs = []
        for i in range(length):
            cur_seq = in_seq[:, : i + 1, :]

            dec_out = self.decoder(
                tgt=cur_seq,  # targets started with CLS token so we have it as the first input during generation
                memory=global_hidden.unsqueeze(1),
            )
            # print(dec_out.size())
            prev_input, cur_output = pred_layers(dec_out[:, -1, :], generation=True)
            in_seq[:, i + 1, :] = prev_input + self.pos_encoding(
                torch.tensor(i + 1, device=global_hidden.device)
            )
            outs.append(cur_output.unsqueeze(1))

        return torch.cat(outs, dim=1)
