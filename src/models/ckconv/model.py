from . import nn as ckconv_nn
import torch
import torch.nn as nn
from .. import preprocessors as prp

class CKCNN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_blocks: int,
        kernelnet_hidden_channels: int,
        kernelnet_activation_function: str,
        kernelnet_norm_type: str,
        dim_linear: int,
        bias: bool,
        omega_0: bool,
        dropout: float,
        weight_dropout: float,
        pool: bool,  # Always False in our experiments.
        out_channels=None,
    ):
        super(CKCNN, self).__init__()

        blocks = []
        for i in range(num_blocks):
            block_in_channels = in_channels if i == 0 else hidden_channels
            block_out_channels = hidden_channels
            if i == num_blocks-1 and out_channels is not None:
                block_out_channels = out_channels
            blocks.append(
                ckconv_nn.CKBlock(
                    block_in_channels,
                    block_out_channels,
                    kernelnet_hidden_channels,
                    kernelnet_activation_function,
                    kernelnet_norm_type,
                    dim_linear,
                    bias,
                    omega_0,
                    dropout,
                    weight_dropout,
                )
            )
            if pool:
                blocks.append(torch.nn.MaxPool1d(kernel_size=2))
        self.backbone = torch.nn.Sequential(*blocks)

    def forward(self, x):
        return self.backbone(x)

class seqImg_CKCNN(CKCNN):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_blocks: int,
        kernelnet_hidden_channels: int,
        kernelnet_activation_function: str,
        kernelnet_norm_type: str,
        dim_linear: int,
        bias: bool,
        omega_0: bool,
        dropout: float,
        weight_dropout: float,
        pool: bool,
    ):
        super().__init__(
            in_channels,
            hidden_channels,
            num_blocks,
            kernelnet_hidden_channels,
            kernelnet_activation_function,
            kernelnet_norm_type,
            dim_linear,
            bias,
            omega_0,
            dropout,
            weight_dropout,
            pool,
        )

        self.finallyr = torch.nn.Linear(
            in_features=hidden_channels, out_features=out_channels
        )
        # Initialize finallyr
        self.finallyr.weight.data.normal_(
            mean=0.0,
            std=0.01,
        )
        self.finallyr.bias.data.fill_(value=0.0)

    def forward(self, x):
        out = self.backbone(x)
        out = self.finallyr(out[:, :, -1])
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out


class CKCONV_Classifier(nn.Module):
    def __init__(self, model_conf, data_conf):
        super().__init__()

        self.model_conf = model_conf
        self.data_conf = data_conf

        ### INPUT SIZE ###
        all_emb_size = self.model_conf.features_emb_dim * len(
            self.data_conf.features.embeddings
        )
        all_numeric_size = len(self.data_conf.features.numeric_values) * (
            self.model_conf.numeric_emb_size if self.model_conf.use_numeric_emb else 1
        )
        self.input_dim = all_emb_size + all_numeric_size

        self.ckconv = seqImg_CKCNN(in_channels = self.input_dim,
                                out_channels=data_conf.num_classes,
                                hidden_channels = model_conf.ckconv.hidden_channels,
                                num_blocks = model_conf.ckconv.num_blocks,
                                kernelnet_hidden_channels = model_conf.ckconv.kernel_hidden_channels,
                                kernelnet_activation_function = model_conf.ckconv.kernel_activation,
                                kernelnet_norm_type = model_conf.ckconv.kernel_norm,
                                dim_linear = 1,
                                bias = True,
                                omega_0 = model_conf.ckconv.omega,
                                dropout = model_conf.ckconv.dropout,
                                weight_dropout = model_conf.ckconv.weight_dropout,
                                pool = False,  # Always False in our experiments.
                                )
        self.processor = prp.FeatureProcessor(
            model_conf=model_conf, data_conf=data_conf
        )

        self.loss_fn = nn.CrossEntropyLoss()
    def forward(self, padded_batch):
        x, time_steps = self.processor(padded_batch)

        outputs = self.ckconv(x.permute(0,2,1))

        # if len(outputs.shape) == 1:
        #     preds = (outputs > 0.0).int()
        # else:
        #     _, preds = torch.max(outputs, 1)

        return outputs

    def loss(self, out, gt):
        loss = self.loss_fn(out, gt[1])

        return {
            "total_loss": loss
        }