import torch
import torch.nn as nn

from ..preprocessors import FeatureProcessor
from .model_utils import MultiTimeAttention, get_normal_kl, get_normal_nll, sample_z


class L2Normalization(nn.Module):
    def __init__(self):
        super(L2Normalization, self).__init__()

    def forward(self, x):
        return x.div(torch.norm(x, dim=1).view(-1, 1))


class SimpleClassifier(nn.Module):
    def __init__(self, model_conf, data_conf):
        super().__init__()

        self.model_conf = model_conf
        self.data_conf = data_conf

        self.processor = FeatureProcessor(model_conf=model_conf, data_conf=data_conf)

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
        self.norm = nn.LayerNorm(self.input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_dim, nhead=self.model_conf.num_heads_enc
        )

        self.enc = nn.TransformerEncoder(
            encoder_layer,
            self.model_conf.num_enc_layers,
            norm=self.norm,
            enable_nested_tensor=True,
            mask_check=True,
        )
        # self.dropout = nn.Dropout(p=0.1)
        self.net = nn.Sequential(
            nn.Linear(
                self.model_conf.classifier_gru_hidden_dim,
                2,  # self.data_conf.num_classes
            ),
            # nn.Sigmoid()
        )
        self.norm2 = nn.LayerNorm(self.model_conf.classifier_gru_hidden_dim)
        # torch.nn.BatchNorm1d(
        #     self.model_conf.classifier_gru_hidden_dim
        # )  # L2Normalization()

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, padded_batch):
        x, time_steps = self.processor(padded_batch)
        # normed = self.norm(x)

        encoded = self.enc(x)
        all_hiddens, _ = self.gru(encoded)
        lens = padded_batch.seq_lens - 1
        last_hidden = self.norm2(all_hiddens[:, lens, :].diagonal().T)
        # last_hidden = all_hiddens[:, lens, :].diagonal().T
        # out, _ = all_hiddens.max(dim=1)
        # print(last_hidden.size())
        return self.net(last_hidden)

    def loss(self, out, gt):
        loss = self.loss_fn(out, gt[1])
        return {"total_loss": loss}


class HalfMoonClassifier(nn.Module):
    def __init__(self, model_conf, data_conf):
        super().__init__()

        self.model_conf = model_conf
        self.data_conf = data_conf

        self.processor = FeatureProcessor(model_conf=model_conf, data_conf=data_conf)

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
