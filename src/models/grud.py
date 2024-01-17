import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import math
import warnings
import itertools
import numbers
import torch.utils.data as utils


def create_beta(feature, times):
    beta = torch.zeros_like(feature)

    for i in range(1, times.size(1)):
        mask = torch.logical_and((feature[:, i - 1] != -1), (feature[:, i - 1] != 0))
        padding_mask = times[:, i] != -1

        beta[:, i] = (
            times[:, i] - times[:, i - 1] + mask * beta[:, i - 1]
        ) * padding_mask

    return beta


class GRUDClassifier(nn.Module):
    def __init__(self, model_conf, data_conf):
        super().__init__()
        self.model_conf = model_conf
        self.data_conf = data_conf

        ### PROCESSORS ###
        self.processor = GRUDFeatureProcessor(
            model_conf=model_conf, data_conf=data_conf
        )

        ### INPUT SIZE ###
        all_emb_size = self.model_conf.features_emb_dim * len(
            self.data_conf.features.embeddings
        )
        all_numeric_size = len(self.data_conf.features.numeric_values) * (
            self.model_conf.numeric_emb_size if self.model_conf.use_numeric_emb else 1
        )
        self.input_dim = all_emb_size + all_numeric_size

        self.encoder = GRUD(
            input_size=self.input_dim,
            hidden_size=self.model_conf.GRUD.hidden_dim,
            output_size=2,
            num_layers=self.model_conf.GRUD.num_layers,
            x_mean=torch.ones(self.input_dim).float(),  # the most fucked up parameter
            bias=True,
            batch_first=True,
            bidirectional=False,
            dropout_type="mloss",
            dropout=self.model_conf.GRUD.dropout,
        )

        self.out_proj = nn.Linear(
            self.model_conf.GRUD.hidden_dim, self.data_conf.num_classes
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, padded_batch):
        x, time_steps, masks, deltas = self.processor(padded_batch, use_norm=False)

        all_hiddens = self.encoder(
            x.permute(0, 2, 1),
            masks.permute(0, 2, 1).float(),
            deltas.permute(0, 2, 1).float(),
        )

        lens = padded_batch.seq_lens - 1
        last_hidden = all_hiddens[:, lens, :].diagonal().T
        y = self.out_proj(last_hidden)

        return y

    def loss(self, out, gt):
        loss = self.loss_fn(out, gt[1])

        return {"total_loss": loss}


class GRUDFeatureProcessor(nn.Module):
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

            self.numeric_norms[name] = nn.Identity()

    def forward(self, padded_batch, use_norm=True):
        numeric_values = []
        categoric_values = []

        numeric_delta = []
        categoric_delta = []

        numeric_masking = []
        categoric_masking = []

        time_steps = padded_batch.payload.get("event_time").float()
        seq_lens = padded_batch.seq_lens
        for key, values in padded_batch.payload.items():
            if key in self.emb_names:
                categoric_values.append(self.embed_layers[key](values.long()))
                categoric_delta.append(
                    create_beta(values, time_steps)
                    .unsqueeze(-1)
                    .repeat(1, 1, self.model_conf.features_emb_dim)
                )
                categoric_masking.append(
                    torch.logical_and((values != -1), (values != 0))
                    .unsqueeze(-1)
                    .repeat(1, 1, self.model_conf.features_emb_dim)
                )
            elif key in self.numeric_names:
                if use_norm:
                    cur_value = self.numeric_norms[key](values.float(), seq_lens)
                else:  # we do not want to use normalization when applying decoder to our sequence
                    # otherwise it would be hard to make true generation and lead to bias in forecasting
                    cur_value = values.float().unsqueeze(-1)

                numeric_values.append(self.numeric_processor[key](cur_value))
                numeric_delta.append(
                    create_beta(values, time_steps)
                    .unsqueeze(-1)
                    .repeat(1, 1, self.model_conf.numeric_emb_size)
                )
                numeric_masking.append(
                    torch.logical_and((values != -1), (values != 0))
                    .unsqueeze(-1)
                    .repeat(1, 1, self.model_conf.numeric_emb_size)
                )

        if len(categoric_values) == 0:
            return torch.cat(numeric_values, dim=-1), time_steps
        if len(numeric_values) == 0:
            return torch.cat(categoric_values, dim=-1), time_steps

        categoric_tensor = torch.cat(categoric_values, dim=-1)
        numeric_tensor = torch.cat(numeric_values, dim=-1)

        numeric_delta_tensor = torch.cat(numeric_delta, dim=-1)
        categoric_delta_tensor = torch.cat(categoric_delta, dim=-1)

        numeric_mask_tensor = torch.cat(numeric_masking, dim=-1)
        categoric_mask_tensor = torch.cat(categoric_masking, dim=-1)
        return (
            torch.cat([categoric_tensor, numeric_tensor], dim=-1),
            time_steps,
            torch.cat([categoric_delta_tensor, numeric_delta_tensor], dim=-1),
            torch.cat([categoric_mask_tensor, numeric_mask_tensor], dim=-1),
        )


class GRUD_cell(torch.nn.Module):
    """
    Implementation of GRUD.
    Inputs: x_mean
            n_smp x 3 x n_channels x len_seq tensor (0: data, 1: mask, 2: deltat)
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=1,
        x_mean=0.0,
        bias=True,
        batch_first=False,
        bidirectional=False,
        dropout_type="mloss",
        dropout=0,
        return_hidden=False,
        device="cpu",
    ):
        super(GRUD_cell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_hidden = (
            return_hidden  # controls the output, True if another GRU-D layer follows
        )

        x_mean = torch.tensor(x_mean, requires_grad=True)
        self.register_buffer("x_mean", x_mean)
        self.bias = bias
        self.batch_first = batch_first
        self.dropout_type = dropout_type
        self.dropout = dropout
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        if (
            not isinstance(dropout, numbers.Number)
            or not 0 <= dropout <= 1
            or isinstance(dropout, bool)
        ):
            raise ValueError(
                "dropout should be a number in range [0, 1] "
                "representing the probability of an element being "
                "zeroed"
            )
        if dropout > 0 and num_layers == 1:
            warnings.warn(
                "dropout option adds dropout after all but last "
                "recurrent layer, so non-zero dropout expects "
                "num_layers greater than 1, but got dropout={} and "
                "num_layers={}".format(dropout, num_layers)
            )

        # set up all the operations that are needed in the forward pass
        self.w_dg_x = torch.nn.Linear(input_size, input_size, bias=True)
        self.w_dg_h = torch.nn.Linear(input_size, hidden_size, bias=True)

        self.w_xz = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.w_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_mz = torch.nn.Linear(input_size, hidden_size, bias=True)

        self.w_xr = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.w_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_mr = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.w_xh = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.w_hh = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_mh = torch.nn.Linear(input_size, hidden_size, bias=True)

        #  self.w_hy = torch.nn.Linear(hidden_size, output_size, bias=True)

        Hidden_State = torch.zeros(self.hidden_size, requires_grad=True)
        # we use buffers because pytorch will take care of pushing them to GPU for us
        self.register_buffer("Hidden_State", Hidden_State)
        self.register_buffer(
            "X_last_obs", torch.zeros(input_size)
        )  # torch.tensor(x_mean) #TODO: what to initialize last observed values with?, also check broadcasting behaviour

        # TODO: check usefulness of everything below here, just copied skeleton

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def check_forward_args(self, input, hidden, batch_sizes):
        is_input_packed = batch_sizes is not None
        expected_input_dim = 2 if is_input_packed else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                "input must have {} dimensions, got {}".format(
                    expected_input_dim, input.dim()
                )
            )
        if self.input_size != input.size(-1):
            raise RuntimeError(
                "input.size(-1) must be equal to input_size. Expected {}, got {}".format(
                    self.input_size, input.size(-1)
                )
            )

        if is_input_packed:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)

        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (
            self.num_layers * num_directions,
            mini_batch,
            self.hidden_size,
        )

        def check_hidden_size(
            hx, expected_hidden_size, msg="Expected hidden size {}, got {}"
        ):
            if tuple(hx.size()) != expected_hidden_size:
                raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

        if self.mode == "LSTM":
            check_hidden_size(
                hidden[0], expected_hidden_size, "Expected hidden[0] size {}, got {}"
            )
            check_hidden_size(
                hidden[1], expected_hidden_size, "Expected hidden[1] size {}, got {}"
            )
        else:
            check_hidden_size(hidden, expected_hidden_size)

    def extra_repr(self):
        s = "{input_size}, {hidden_size}"
        if self.num_layers != 1:
            s += ", num_layers={num_layers}"
        if self.bias is not True:
            s += ", bias={bias}"
        if self.batch_first is not False:
            s += ", batch_first={batch_first}"
        if self.dropout != 0:
            s += ", dropout={dropout}"
        if self.bidirectional is not False:
            s += ", bidirectional={bidirectional}"
        return s.format(**self.__dict__)

    @property
    def _flat_weights(self):
        return list(self._parameters.values())

    def forward(self, X, Mask, Delta):
        # input.size = (3, 33,49) : num_input or num_hidden, num_layer or step
        # X = torch.squeeze(input[0]) # .size = (33,49)
        # Mask = torch.squeeze(input[1]) # .size = (33,49)
        # Delta = torch.squeeze(input[2]) # .size = (33,49)
        # X = input[:,0,:,:]
        # Mask = input[:,1,:,:]
        # Delta = input[:,2,:,:]

        step_size = X.size(1)  # 49
        # print('step size : ', step_size)

        output = None
        # h = Hidden_State
        h = getattr(self, "Hidden_State")
        # felix - buffer system from newer pytorch version
        x_mean = getattr(self, "x_mean")
        x_last_obsv = getattr(self, "X_last_obs")

        device = next(self.parameters()).device
        output_tensor = torch.empty(
            [X.size()[0], X.size()[2], self.output_size], dtype=X.dtype, device=device
        )
        hidden_tensor = torch.empty(
            X.size()[0], X.size()[2], self.hidden_size, dtype=X.dtype, device=device
        )

        # iterate over seq
        for timestep in range(X.size()[2]):
            # x = torch.squeeze(X[:,layer:layer+1])
            # m = torch.squeeze(Mask[:,layer:layer+1])
            # d = torch.squeeze(Delta[:,layer:layer+1])
            x = torch.squeeze(X[:, :, timestep])
            m = torch.squeeze(Mask[:, :, timestep])
            d = torch.squeeze(Delta[:, :, timestep])

            # (4)
            gamma_x = torch.exp(-1 * torch.nn.functional.relu(self.w_dg_x(d)))
            gamma_h = torch.exp(-1 * torch.nn.functional.relu(self.w_dg_h(d)))

            # (5)
            # standard mult handles case correctly, this should work - maybe broadcast x_mean, seems to be taking care of that anyway

            x_last_obsv = torch.where(m > 0, x, x_last_obsv)

            x = m * x + (1 - m) * (gamma_x * x + (1 - gamma_x) * x_mean)
            x = m * x + (1 - m) * (gamma_x * x_last_obsv + (1 - gamma_x) * x_mean)

            # (6)
            if self.dropout == 0:
                h = gamma_h * h
                z = torch.sigmoid(self.w_xz(x) + self.w_hz(h) + self.w_mz(m))
                r = torch.sigmoid(self.w_xr(x) + self.w_hr(h) + self.w_mr(m))

                h_tilde = torch.tanh(self.w_xh(x) + self.w_hh(r * h) + self.w_mh(m))

                h = (1 - z) * h + z * h_tilde

            # TODO: not adapted yet
            elif self.dropout_type == "Moon":
                """
                RNNDROP: a novel dropout for rnn in asr(2015)
                """
                h = gamma_h * h

                z = torch.sigmoid((w_xz * x + w_hz * h + w_mz * m + b_z))
                r = torch.sigmoid((w_xr * x + w_hr * h + w_mr * m + b_r))

                h_tilde = torch.tanh((w_xh * x + w_hh * (r * h) + w_mh * m + b_h))

                h = (1 - z) * h + z * h_tilde
                dropout = torch.nn.Dropout(p=self.dropout)
                h = dropout(h)

            elif self.dropout_type == "Gal":
                """
                A Theoretically grounded application of dropout in recurrent neural networks(2015)
                """
                dropout = torch.nn.Dropout(p=self.dropout)
                h = dropout(h)

                h = gamma_h * h

                z = torch.sigmoid((w_xz * x + w_hz * h + w_mz * m + b_z))
                r = torch.sigmoid((w_xr * x + w_hr * h + w_mr * m + b_r))
                h_tilde = torch.tanh((w_xh * x + w_hh * (r * h) + w_mh * m + b_h))

                h = (1 - z) * h + z * h_tilde

            elif self.dropout_type == "mloss":
                """
                recurrent dropout without memory loss arXiv 1603.05118
                g = h_tilde, p = the probability to not drop a neuron
                """
                h = gamma_h * h
                z = torch.sigmoid(self.w_xz(x) + self.w_hz(h) + self.w_mz(m))
                r = torch.sigmoid(self.w_xr(x) + self.w_hr(h) + self.w_mr(m))

                dropout = torch.nn.Dropout(p=self.dropout)
                h_tilde = torch.tanh(self.w_xh(x) + self.w_hh(r * h) + self.w_mh(m))

                h = (1 - z) * h + z * h_tilde
                #######

            else:
                h = gamma_h * h

                z = torch.sigmoid((w_xz * x + w_hz * h + w_mz * m + b_z))
                r = torch.sigmoid((w_xr * x + w_hr * h + w_mr * m + b_r))
                h_tilde = torch.tanh((w_xh * x + w_hh * (r * h) + w_mh * m + b_h))

                h = (1 - z) * h + z * h_tilde

            # step_output = self.w_hy(h)
            # step_output = torch.sigmoid(step_output)
            #  output_tensor[:,timestep,:] = h
            hidden_tensor[:, timestep, :] = h

        # if self.return_hidden:
        # when i want to stack GRU-Ds, need to put the tensor back together
        # output = torch.stack([hidden_tensor,Mask,Delta], dim=1)

        # output = output_tensor, hidden_tensor
        output = hidden_tensor
        # else:
        #    output = output_tensor
        return output


class GRUD(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=1,
        x_mean=0,
        bias=True,
        batch_first=False,
        bidirectional=False,
        dropout_type="mloss",
        dropout=0,
    ):
        super().__init__()

        self.gru_d = GRUD_cell(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout=dropout,
            dropout_type=dropout_type,
            x_mean=x_mean,
        )
        #  self.hidden_to_output = torch.nn.Linear(hidden_size, output_size, bias=True)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        if self.num_layers > 1:
            # (batch, seq, feature)
            self.gru_layers = torch.nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                batch_first=True,
                num_layers=self.num_layers - 1,
                dropout=dropout,
            )

    def initialize_hidden(self, batch_size):
        device = next(self.parameters()).device
        # The hidden state at the start are all zeros
        return torch.zeros(
            self.num_layers - 1, batch_size, self.hidden_size, device=device
        )

    def forward(self, x, masks, deltas):
        # pass through GRU-D
        #  output, hidden = self.gru_d(x, masks, deltas)
        hidden = self.gru_d(x, masks, deltas)
        # print(self.gru_d.return_hidden)
        # output = self.gru_d(input)
        # print(output.size())

        # batch_size, n_hidden, n_timesteps

        if self.num_layers > 1:
            # TODO remove init hidden, not necessary, auto init works fine
            init_hidden = self.initialize_hidden(hidden.size()[0])

            output, hidden = self.gru_layers(hidden)  # , init_hidden)

            return output
        # output = self.hidden_to_output(output)
        #  output = torch.sigmoid(output)

        # print("final output size passed as model result")
        # print(output.size())
        return hidden
