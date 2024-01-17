import torch
import torch.nn as nn
from . import preprocessors as prp
from ..trainers.losses import get_loss
from .model_utils import L2Normalization, FeatureMixer


class BaseMixin(nn.Module):
    def __init__(self, model_conf, data_conf):
        super().__init__()
        self.model_conf = model_conf
        self.data_conf = data_conf

        ### PROCESSORS ###
        self.processor = prp.FeatureProcessor(
            model_conf=model_conf, data_conf=data_conf
        )
        # TIME PROCESSOR DISABLED !!! ###
        time_preproc = self.model_conf.get("time_preproc", None)
        assert time_preproc in [None, "Identity"], "TIME PROCESSOR DISABLED"
        # self.time_processor = getattr(prp, self.model_conf.time_preproc)(
        #     self.model_conf.num_time_blocks, model_conf.device
        # )
        self.time_processor = prp.Identity()
        self.time_encoder = prp.TimeEncoder(data_conf=data_conf, model_conf=model_conf)

        ### INPUT SIZE ###
        all_emb_size = self.model_conf.features_emb_dim * len(
            self.data_conf.features.embeddings
        )
        all_numeric_size = len(self.data_conf.features.numeric_values) * (
            self.model_conf.numeric_emb_size if self.model_conf.use_numeric_emb else 1
        )
        self.input_dim = all_emb_size + all_numeric_size + self.model_conf.use_deltas
        if self.model_conf.time_embedding and self.model_conf.use_deltas:
            self.input_dim += self.model_conf.time_embedding * 2 - 1

        ### MIXER ###
        if self.model_conf.encoder_feature_mixer:
            assert self.model_conf.use_numeric_emb
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

        ### NORMS ###
        self.pre_encoder_norm = getattr(nn, self.model_conf.pre_encoder_norm)(
            self.input_dim
        )
        self.post_encoder_norm = getattr(nn, self.model_conf.post_encoder_norm)(
            self.model_conf.encoder_hidden
        )
        self.encoder_norm = getattr(nn, self.model_conf.encoder_norm)(self.input_dim)

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

        ### DROPOUT ###
        self.after_enc_dropout = nn.Dropout1d(self.model_conf.after_enc_dropout)

        ### OUT PROJECTION ###
        self.out_linear = getattr(nn, self.model_conf.predict_head)(
            self.model_conf.encoder_hidden, self.data_conf.num_classes
        )

        ### LOSS ###

        self.loss_fn = get_loss(self.model_conf)

    def loss(self, out, gt):
        if self.model_conf.predict_head == "Identity":
            loss = self.loss_fn(out, gt[0])
        else:
            if self.model_conf.loss.loss_fn == "MSE":
                loss = self.loss_fn(out.squeeze(-1), gt[1].float())
            else:
                loss = self.loss_fn(out, gt[1])

        # if self.model_conf.time_preproc == "MultiTimeSummator":
        #     # logp = torch.log(self.time_processor.softmaxed_weights)
        #     # entropy_term = torch.sum(-self.time_processor.softmaxed_weights * logp)
        #     entropy_term = torch.tensor(0)
        # else:
        #     entropy_term = torch.tensor(0)
        return {"total_loss": loss}


class GRUClassifier(BaseMixin):
    def __init__(self, model_conf, data_conf):
        super().__init__(model_conf, data_conf)

        self.encoder = nn.GRU(
            self.input_dim,
            self.model_conf.encoder_hidden,
            batch_first=True,
            num_layers=self.model_conf.encoder_num_layers,
        )

    def forward(self, padded_batch):
        x, time_steps = self.processor(padded_batch)
        x, time_steps = self.time_processor(x, time_steps)

        if self.model_conf.time_embedding:
            x = self.time_encoder(x, time_steps)
            x = self.encoder_feature_mixer(x)
        else:
            x = self.encoder_feature_mixer(x)
            x = self.time_encoder(x, time_steps)

        encoded = self.preENC_TR(x)

        all_hiddens, hn = self.encoder(self.pre_encoder_norm(encoded))
        if not self.model_conf.get("time_preproc", None):
            lens = padded_batch.seq_lens - 1
            last_hidden = self.post_encoder_norm(all_hiddens[:, lens, :].diagonal().T)
        else:
            last_hidden = self.post_encoder_norm(hn.squeeze(0))

        y = self.out_linear(last_hidden)

        return y


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


class ConvClassifier(BaseMixin):
    def __init__(self, model_conf, data_conf):
        super().__init__(model_conf, data_conf)

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

    def forward(self, padded_batch):
        x, time_steps = self.processor(padded_batch)

        out_conv = self.conv_net(x)
        max_out, _ = torch.max(out_conv, dim=1)
        return self.out_linear(self.pre_gru_norm(max_out))


class ConvGRUClassifier(BaseMixin):
    def __init__(self, model_conf, data_conf):
        super().__init__(model_conf, data_conf)

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
            self.input_dim, self.model_conf.encoder_hidden, batch_first=True
        )
        self.out_linear = nn.Linear(
            self.model_conf.encoder_hidden, self.data_conf.num_classes
        )

        self.act = getattr(nn, self.model_conf.activation)()

    def forward(self, padded_batch):
        x, time_steps = self.processor(padded_batch)
        x, time_steps = self.time_processor(x, time_steps)

        out_conv = self.conv_net(x)

        out_conv = self.pre_gru_norm(out_conv)
        all_hiddens, h_n = self.gru(out_conv)
        last_hidden = self.post_gru_norm(h_n.squeeze(0))
        return self.out_linear(last_hidden)
