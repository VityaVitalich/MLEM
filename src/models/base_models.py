import torch
import torch.nn as nn
from . import preprocessors as prp


class L2Normalization(nn.Module):
    def __init__(self):
        super(L2Normalization, self).__init__()

    def forward(self, x):
        return x.div(torch.norm(x, dim=1).view(-1, 1))


class GRUClassifier(nn.Module):
    def __init__(self, model_conf, data_conf):
        super().__init__()

        self.model_conf = model_conf
        self.data_conf = data_conf

        self.processor = prp.FeatureProcessor(
            model_conf=model_conf, data_conf=data_conf
        )
        self.time_processor = getattr(prp, self.model_conf.time_preproc)(
            self.model_conf.num_time_blocks, model_conf.device
        )

        all_emb_size = self.model_conf.features_emb_dim * len(
            self.data_conf.features.embeddings
        )
        all_numeric_size = len(self.data_conf.features.numeric_values)
        self.input_dim = all_emb_size + all_numeric_size

        self.gru = nn.GRU(
            self.input_dim,
            self.model_conf.classifier_gru_hidden_dim,
            batch_first=True,
        )
        self.out_linear = nn.Sequential(
            nn.Linear(
                self.model_conf.classifier_gru_hidden_dim, self.data_conf.num_classes
            )
        )
        self.pre_gru_norm = getattr(nn, self.model_conf.pre_gru_norm)(self.input_dim)
        self.post_gru_norm = getattr(nn, self.model_conf.post_gru_norm)(
            self.model_conf.classifier_gru_hidden_dim
        )
        self.encoder_norm = getattr(nn, self.model_conf.encoder_norm)(self.input_dim)

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
        self.after_enc_dropout = nn.Dropout(self.model_conf.after_enc_dropout)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, padded_batch):
        x, time_steps = self.processor(padded_batch)
        x, time_steps = self.time_processor(x, time_steps)

        encoded = self.encoder(x)

        all_hiddens, hn = self.gru(self.pre_gru_norm(self.after_enc_dropout(encoded)))
        if self.model_conf.time_preproc == "Identity":
            lens = padded_batch.seq_lens - 1
            last_hidden = self.post_gru_norm(all_hiddens[:, lens, :].diagonal().T)
        else:
            last_hidden = self.post_gru_norm(hn.squeeze(0))

        return self.out_linear(last_hidden)

    def loss(self, out, gt):
        loss = self.loss_fn(out, gt[1])
        return {"total_loss": loss}


class ConvStack(nn.Module):
    def __init__(self, model_conf, data_conf, input_dim):
        super().__init__()
        self.model_conf = model_conf
        self.data_conf = data_conf

        self.input_dim = input_dim

        res_dim = self.model_conf.conv.out_channels * (
            len(self.model_conf.conv.kernels) * len(self.model_conf.conv.dilations)
        )

        self.convs = nn.ModuleList()
        for k in self.model_conf.conv.kernels:
            for d in self.model_conf.conv.dilations:
                p = d * (k - 1) // 2
                self.convs.append(
                    nn.Conv1d(
                        self.input_dim,
                        self.model_conf.conv.out_channels,
                        kernel_size=k,
                        padding=p,
                        dilation=d,
                    )
                )

        self.proj = getattr(nn, self.model_conf.conv.proj)(res_dim, self.input_dim)
        self.act = getattr(nn, self.model_conf.activation)()

    def forward(self, x):
        """
        x = [BS, L, D]
        """
        out_conv = torch.concat(
            [conv(x.transpose(1, 2)) for conv in self.convs], dim=1
        ).transpose(1, 2)
        out = self.proj(self.act(out_conv))
        return self.act(out)


class ConvClassifier(nn.Module):
    def __init__(self, model_conf, data_conf):
        super().__init__()

        self.model_conf = model_conf
        self.data_conf = data_conf

        self.processor = prp.FeatureProcessor(
            model_conf=model_conf, data_conf=data_conf
        )
        self.time_processor = getattr(prp, self.model_conf.time_preproc)(
            self.model_conf.num_time_blocks, model_conf.device
        )

        all_emb_size = self.model_conf.features_emb_dim * len(
            self.data_conf.features.embeddings
        )
        all_numeric_size = len(self.data_conf.features.numeric_values)
        self.input_dim = all_emb_size + all_numeric_size
        res_dim = self.model_conf.conv.out_channels * (
            len(self.model_conf.conv.kernels) * len(self.model_conf.conv.dilations)
        )

        self.pre_gru_norm = getattr(nn, self.model_conf.pre_gru_norm)(self.input_dim)
        self.post_gru_norm = getattr(nn, self.model_conf.post_gru_norm)(
            self.model_conf.classifier_gru_hidden_dim
        )
        self.act = getattr(nn, self.model_conf.activation)()

        self.conv_stacks = []
        for i in range(self.model_conf.conv.num_stacks):
            self.conv_stacks.append(
                ConvStack(
                    model_conf=self.model_conf,
                    data_conf=self.data_conf,
                    input_dim=self.input_dim,
                )
            )

        self.conv_net = nn.Sequential(*self.conv_stacks)

        self.out_linear = nn.Linear(self.input_dim, self.data_conf.num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, padded_batch):
        x, time_steps = self.processor(padded_batch)

        out_conv = self.conv_net(x)
        max_out, _ = torch.max(out_conv, dim=1)
        return self.out_linear(self.pre_gru_norm(max_out))

    def loss(self, out, gt):
        loss = self.loss_fn(out, gt[1])
        return {"total_loss": loss}


class ConvGRUClassifier(nn.Module):
    def __init__(self, model_conf, data_conf):
        super().__init__()

        self.model_conf = model_conf
        self.data_conf = data_conf

        self.processor = prp.FeatureProcessor(
            model_conf=model_conf, data_conf=data_conf
        )

        self.time_processor = getattr(prp, self.model_conf.time_preproc)(
            self.model_conf.num_time_blocks, model_conf.device
        )

        all_emb_size = self.model_conf.features_emb_dim * len(
            self.data_conf.features.embeddings
        )
        all_numeric_size = len(self.data_conf.features.numeric_values)
        self.input_dim = all_emb_size + all_numeric_size

        self.conv_stacks = []
        for i in range(self.model_conf.conv.num_stacks):
            self.conv_stacks.append(
                ConvStack(
                    model_conf=self.model_conf,
                    data_conf=self.data_conf,
                    input_dim=self.input_dim,
                )
            )

        self.conv_net = nn.Sequential(*self.conv_stacks)

        self.gru = nn.GRU(
            self.input_dim, self.model_conf.classifier_gru_hidden_dim, batch_first=True
        )
        self.out_linear = nn.Linear(
            self.model_conf.classifier_gru_hidden_dim, self.data_conf.num_classes
        )

        self.pre_gru_norm = getattr(nn, self.model_conf.pre_gru_norm)(self.input_dim)
        self.post_gru_norm = getattr(nn, self.model_conf.post_gru_norm)(
            self.model_conf.classifier_gru_hidden_dim
        )
        self.act = getattr(nn, self.model_conf.activation)()

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, padded_batch):
        x, time_steps = self.processor(padded_batch)
        x, time_steps = self.time_processor(x, time_steps)

        out_conv = self.conv_net(x)

        out_conv = self.pre_gru_norm(out_conv)
        all_hiddens, h_n = self.gru(out_conv)
        last_hidden = self.post_gru_norm(h_n.squeeze(0))
        return self.out_linear(last_hidden)

    def loss(self, out, gt):
        loss = self.loss_fn(out, gt[1])
        return {"total_loss": loss}


class HalfMoonClassifier(nn.Module):
    def __init__(self, model_conf, data_conf):
        super().__init__()

        self.model_conf = model_conf
        self.data_conf = data_conf

        self.processor = prp.FeatureProcessor(
            model_conf=model_conf, data_conf=data_conf
        )

        all_emb_size = self.model_conf.features_emb_dim * len(
            self.data_conf.features.embeddings
        )
        all_numeric_size = len(self.data_conf.features.numeric_values)
        self.input_dim = all_emb_size + all_numeric_size

        self.net = nn.Sequential(
            torch.nn.Linear(2, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 2),
        )
        # self.norm = nn.LayerNorm(self.input_dim)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, padded_batch):
        x, time_steps = self.processor(padded_batch)
        # print(x.size())
        # normed = self.norm(x)
        return self.net(x).squeeze(1)

    def loss(self, out, gt):
        # print(out.size(), gt[1].size())
        loss = self.loss_fn(out, gt[1])
        return {"total_loss": loss}
