import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from . import preprocessors as prp
from .gen_models import NumericalFeatureProjector


class EmbeddingPredictor(nn.Module):
    def __init__(self, model_conf, data_conf):
        super().__init__()
        self.model_conf = model_conf
        self.data_conf = data_conf

        self.criterion = nn.CrossEntropyLoss(
            reduction="none", ignore_index=0, label_smoothing=0.15
        )

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
                shifted_labels = padded_batch.payload[name].long()  # [:, 1:]
                embed_losses[name] = (
                    self.criterion(dist.permute(0, 2, 1), shifted_labels)
                    .sum(dim=1)
                    .mean()
                )

        return embed_losses


def random_generator(batch_size, z_dim, T_mb, max_seq_len):
    """Random vector generation.

    Args:
      - batch_size: size of the random vector
      - z_dim: dimension of random vector
      - T_mb: time information for the random vector
      - max_seq_len: maximum sequence length

    Returns:
      - Z_mb: generated random vector
    """
    Z_mb = list()
    for i in range(batch_size):
        temp = np.zeros([max_seq_len, z_dim])
        temp_Z = np.random.uniform(0.0, 1, [T_mb[i], z_dim])
        temp[: T_mb[i], :] = temp_Z
        Z_mb.append(temp)
    return torch.tensor(np.stack(Z_mb)).float()


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
    elif classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("Norm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)


class Encoder(nn.Module):
    """Embedding network between original feature space to latent space.

    Args:
      - input: input time-series features. (L, N, X) = (24, ?, 6)
      - h3: (num_layers, N, H). [3, ?, 24]

    Returns:
      - H: embeddings
    """

    def __init__(self, input_size, hidden_rnn, num_layers):
        super(Encoder, self).__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_rnn,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_rnn, hidden_rnn)
        self.sigmoid = nn.Sigmoid()
        # в оригинале стремная инициализация весов
        self.apply(_weights_init)

    def forward(self, x, sigmoid=True):
        e_outputs, _ = self.rnn(x)
        H = self.fc(e_outputs)
        if sigmoid:
            H = self.sigmoid(H)
        return H


class Recovery(nn.Module):
    """Recovery network from latent space to original space.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - X_tilde: recovered data
    """

    def __init__(self, input_size, hidden_rnn, num_layers):
        super(Recovery, self).__init__()
        self.rnn = nn.GRU(
            input_size=hidden_rnn,
            hidden_size=input_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        r_outputs, _ = self.rnn(input)
        X_tilde = self.fc(r_outputs)
        if sigmoid:
            X_tilde = self.sigmoid(X_tilde)
        return X_tilde


class Generator(nn.Module):
    """Generator function: Generate time-series data in latent space.

    Args:
      - Z: random variables
      - T: input time information

    Returns:
      - E: generated embedding
    """

    def __init__(self, input_size, hidden_rnn, num_layers):
        super(Generator, self).__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_rnn,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_rnn, hidden_rnn)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        g_outputs, _ = self.rnn(input)
        E = self.fc(g_outputs)
        if sigmoid:
            E = self.sigmoid(E)
        return E


class Supervisor(nn.Module):
    """Generate next sequence using the previous sequence.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - S: generated sequence based on the latent representations generated by the generator
    """

    def __init__(self, hidden_rnn, num_layers):
        super(Supervisor, self).__init__()
        self.rnn = nn.GRU(
            input_size=hidden_rnn,
            hidden_size=hidden_rnn,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_rnn, hidden_rnn)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        s_outputs, _ = self.rnn(input)
        S = self.fc(s_outputs)
        if sigmoid:
            S = self.sigmoid(S)
        return S


class Discriminator(nn.Module):
    """Discriminate the original and synthetic time-series data.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - Y_hat: classification results between original and synthetic time-series
    """

    def __init__(self, hidden_rnn, num_layers):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(
            input_size=hidden_rnn,
            hidden_size=hidden_rnn,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_rnn, 2)
        self.apply(_weights_init)

    def forward(self, input):
        d_outputs, _ = self.rnn(input)
        Y_hat = self.fc(d_outputs)
        return Y_hat


class TG(nn.Module):
    def __init__(self, data_conf, model_conf):
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
        assert self.model_conf.time_embedding == 0

        self.encoder = Encoder(
            input_size=self.input_dim,
            hidden_rnn=self.model_conf.timegan.rnn_hidden,
            num_layers=self.model_conf.timegan.num_layers,
        )
        self.decoder = Recovery(
            input_size=self.input_dim,
            hidden_rnn=self.model_conf.timegan.rnn_hidden,
            num_layers=self.model_conf.timegan.num_layers,
        )
        self.supervisor = Supervisor(
            hidden_rnn=self.model_conf.timegan.rnn_hidden,
            num_layers=self.model_conf.timegan.num_layers,
        )
        self.generator = Generator(
            input_size=self.input_dim,
            hidden_rnn=self.model_conf.timegan.rnn_hidden,
            num_layers=self.model_conf.timegan.num_layers,
        )
        self.discriminator = Discriminator(
            hidden_rnn=self.model_conf.timegan.rnn_hidden,
            num_layers=self.model_conf.timegan.num_layers,
        )

        self.gamma = self.model_conf.timegan.gamma

        # Predictors
        self.embedding_predictor = EmbeddingPredictor(
            model_conf=self.model_conf, data_conf=self.data_conf
        )
        self.numeric_projector = NumericalFeatureProjector(
            model_conf=self.model_conf, data_conf=self.data_conf
        )
        self.mse_fn = torch.nn.MSELoss(reduction="none")
        # self.ce_fn = torch.nn.CrossEntropyLoss(
        #     reduction="mean", ignore_index=0, label_smoothing=0.15
        # )

    def numerical_loss(self, pred, input_batch):
        # MSE
        total_mse_loss = 0
        for key, values in input_batch.payload.items():
            if key in self.processor.numeric_names:
                gt_val = values.float()
                pred_val = pred[key].squeeze(-1)

                mse_loss = self.mse_fn(
                    gt_val,
                    pred_val,
                )
                mask = gt_val != 0
                masked_mse = mse_loss * mask
                total_mse_loss += (
                    masked_mse.sum(dim=1)  # / (mask != 0).sum(dim=1)
                ).mean()

        return total_mse_loss

    def delta_mse_loss(self, time_steps, pred_delta):
        # DELTA MSE
        if self.model_conf.use_deltas:
            gt_delta = time_steps.diff(1)
            if self.model_conf.use_log_delta:
                gt_delta = torch.log(gt_delta + 1e-10)
            delta_mse = self.mse_fn(gt_delta, pred_delta[:, :-1])
            mask = time_steps != -1

            delta_masked = delta_mse * mask[:, :-1]
            delta_mse = delta_masked.sum() / (mask != 0).sum()
        else:
            delta_mse = torch.tensor(0)

        return delta_mse

    def e_loss(self, decoded, padded_batch):
        emb_dist = self.embedding_predictor(decoded)
        cross_entropy_losses = self.embedding_predictor.loss(emb_dist, padded_batch)
        total_ce_loss = torch.sum(
            torch.cat([value.unsqueeze(0) for _, value in cross_entropy_losses.items()])
        )

        mse_loss = self.numerical_loss(self.numeric_projector(decoded), padded_batch)
        delta_loss = self.delta_mse_loss(
            padded_batch.payload["event_time"].float(), decoded[:, :, -1]
        )
        return total_ce_loss + mse_loss + self.model_conf.delta_weight * delta_loss

    def train_embedder(self, padded_batch):
        x, time_steps = self.processor(padded_batch)
        x = self.time_encoder(x, time_steps)

        latens = self.encoder(x)
        decoded = self.decoder(latens)

        lens = padded_batch.seq_lens - 1
        global_hidden = latens[:, lens, :].diagonal().T

        total_loss = self.e_loss(decoded, padded_batch)

        return global_hidden, total_loss

    def train_generator(self, padded_batch):
        x, time_steps = self.processor(padded_batch)
        x = self.time_encoder(x, time_steps)

        bs, l, d = x.size()
        Z = random_generator(bs, d, [l] * bs, l)
        gen_E = self.generator(Z.to(self.model_conf.device))

        gen_latens = self.supervisor(gen_E)
        H = self.encoder(x)
        H_hat_supervised = self.supervisor(H)

        mse = (
            F.mse_loss(H_hat_supervised, gen_latens, reduction="none")
            .sum(dim=[1, 2])
            .mean()
        )
        # mse = F.mse_loss(H[:,1:,:], H_hat_supervised[:,:-1, :], reduction="none").sum(dim=[1, 2]).mean()
        return mse

    def train_joint(self, padded_batch):
        x, time_steps = self.processor(padded_batch)
        x = self.time_encoder(x, time_steps)

        bs, l, d = x.size()
        Z = random_generator(bs, d, [l] * bs, l)
        gen_latens = self.supervisor(self.generator(Z.to(self.model_conf.device)))
        latens = self.encoder(x)
        H_hat_supervised = self.supervisor(latens)
        decoded = self.decoder(latens)
        gen_decoded = self.decoder(gen_latens)

        y_fake = self.discriminator(gen_latens)
        g_loss_u = (
            F.cross_entropy(
                y_fake.permute(0, 2, 1),
                torch.ones_like(y_fake)[:, :, 0].long(),
                reduction="none",
            )
            .sum(dim=1)
            .mean()
        )
        g_loss_s = (
            # F.mse_loss(latens[:,1:,:], H_hat_supervised[:,:-1, :], reduction="none").sum(dim=[1, 2]).mean()
            F.mse_loss(H_hat_supervised, gen_latens, reduction="none")
            .sum(dim=[1, 2])
            .mean()
        )

        G_loss_V1 = torch.mean(
            torch.abs(
                (torch.std(gen_decoded, dim=0)) + 1e-6 - (torch.std(x, dim=0) + 1e-6)
            )
        )
        G_loss_V2 = torch.mean(
            torch.abs((torch.mean(gen_decoded, dim=0) - (torch.mean(x, dim=0))))
        )
        g_loss_v = G_loss_V1 + G_loss_V2  # var_loss + mean_loss

        e_loss = self.e_loss(decoded, padded_batch)
        e_loss = 10 * torch.sqrt(e_loss)
        e_loss = e_loss + 0.1 * g_loss_s

        return g_loss_u, g_loss_s, g_loss_v, e_loss

    def train_discriminator(self, padded_batch):
        x, time_steps = self.processor(padded_batch)
        x = self.time_encoder(x, time_steps)

        bs, l, d = x.size()
        Z = random_generator(bs, d, [l] * bs, l)
        e_gen = self.generator(Z.to(self.model_conf.device))
        gen_latens = self.supervisor(e_gen)
        latens = self.encoder(x)

        y_fake = self.discriminator(gen_latens)
        y_fake_e = self.discriminator(e_gen)
        y_real = self.discriminator(latens)
        D_loss_fake = (
            F.cross_entropy(
                y_fake.permute(0, 2, 1),
                torch.zeros_like(y_fake)[:, :, 0].long(),
                reduction="none",
            )
            .sum(dim=1)
            .mean()
        )
        D_loss_real = (
            F.cross_entropy(
                y_real.permute(0, 2, 1),
                torch.ones_like(y_real)[:, :, 0].long(),
                reduction="none",
            )
            .sum(dim=1)
            .mean()
        )
        D_loss_fake_e = (
            F.cross_entropy(
                y_fake_e.permute(0, 2, 1),
                torch.zeros_like(y_fake_e)[:, :, 0].long(),
                reduction="none",
            )
            .sum(dim=1)
            .mean()
        )
        D_loss = D_loss_real + D_loss_fake + self.gamma * D_loss_fake_e

        if D_loss.item() <= 0.15:
            D_loss = torch.tensor(0)

        return D_loss

    def reconstruct(self, padded_batch):
        x, time_steps = self.processor(padded_batch)
        x = self.time_encoder(x, time_steps)

        latens = self.encoder(x)
        decoded = self.decoder(latens)

        pred = self.embedding_predictor(decoded)
        pred.update(self.numeric_projector(decoded))

        if self.model_conf.use_deltas:
            pred["delta"] = torch.abs(decoded[:, :, -1].squeeze(-1))

        out = {"pred": pred, "time_steps": time_steps}
        return out

    def generate(self, padded_batch):
        x, time_steps = self.processor(padded_batch)
        x = self.time_encoder(x, time_steps)

        bs, l, d = x.size()
        Z = random_generator(bs, d, [l] * bs, l)
        gen_latens = self.supervisor(self.generator(Z.to(self.model_conf.device)))
        gen_decoded = self.decoder(gen_latens)

        pred = self.embedding_predictor(gen_decoded)
        pred.update(self.numeric_projector(gen_decoded))
        if self.model_conf.use_deltas:
            pred["delta"] = torch.abs(gen_decoded[:, :, -1].squeeze(-1))

        out = {"pred": pred, "time_steps": time_steps}
        return out
