import torch
import torch.nn as nn

from ..preprocessors import FeatureProcessor
from ..model_utils import MultiTimeAttention, get_normal_kl, get_normal_nll, sample_z


class EncMtanRnn(nn.Module):
    def __init__(
        self,
        input_dim,
        query,
        latent_dim=2,
        nhidden=16,
        embed_time=16,
        num_heads=1,
        linear_hidden_dim=50,
        num_time_emb=3,
        device="cuda",
    ):
        super().__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.linear_hidden_dim = linear_hidden_dim
        self.num_time_emb = num_time_emb
        self.att = MultiTimeAttention(
            input_dim, nhidden, embed_time, num_heads, num_time_emb
        )
        self.gru_rnn = nn.GRU(nhidden, nhidden, bidirectional=True, batch_first=True)
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(2 * nhidden, self.linear_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.linear_hidden_dim, latent_dim * 2),
        )

        periodic = []
        linear_time = []
        for _ in range(self.num_time_emb):
            periodic.append(nn.Linear(1, embed_time - 1))
            linear_time.append(nn.Linear(1, 1))
        self.periodic = nn.ModuleList(periodic)
        self.linear_time = nn.ModuleList(linear_time)

    def learn_time_embedding(self, tt, emb_index):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic[emb_index](tt))
        out1 = self.linear_time[emb_index](tt)
        return torch.cat([out1, out2], -1)

    def forward(self, x, time_steps):
        time_steps = time_steps.cpu()

        keys = []
        querys = []
        for emb_index in range(self.num_time_emb):
            keys.append(self.learn_time_embedding(time_steps, emb_index).unsqueeze(1))
            # first unsqueezed to match batch dimension of time steps
            # second unsqueeze to keys and querys to concat over num_time_emb
            querys.append(
                self.learn_time_embedding(self.query.unsqueeze(0), emb_index).unsqueeze(
                    1
                )
            )
        keys = torch.cat(keys, dim=1).to(self.device)
        querys = torch.cat(querys, dim=1).to(self.device)

        out = self.att(querys, keys, x, mask=None)
        out, _ = self.gru_rnn(out)
        out = self.hiddens_to_z0(out)
        return out


class EncMtanRnnClassification(nn.Module):
    def __init__(
        self,
        input_dim,
        query,
        latent_dim=2,
        nhidden=16,
        embed_time=16,
        num_heads=1,
        linear_hidden_dim=50,
        num_time_emb=3,
        device="cuda",
    ):
        super().__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.linear_hidden_dim = linear_hidden_dim
        self.num_time_emb = num_time_emb
        self.att = MultiTimeAttention(
            input_dim, nhidden, embed_time, num_heads, num_time_emb
        )

        periodic = []
        linear_time = []
        for _ in range(self.num_time_emb):
            periodic.append(nn.Linear(1, embed_time - 1))
            linear_time.append(nn.Linear(1, 1))
        self.periodic = nn.ModuleList(periodic)
        self.linear_time = nn.ModuleList(linear_time)

    def learn_time_embedding(self, tt, emb_index):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic[emb_index](tt))
        out1 = self.linear_time[emb_index](tt)
        return torch.cat([out1, out2], -1)

    def forward(self, x, time_steps):
        time_steps = time_steps.cpu()

        keys = []
        querys = []
        for emb_index in range(self.num_time_emb):
            keys.append(self.learn_time_embedding(time_steps, emb_index).unsqueeze(1))
            # first unsqueezed to match batch dimension of time steps
            # second unsqueeze to keys and querys to concat over num_time_emb
            querys.append(
                self.learn_time_embedding(self.query.unsqueeze(0), emb_index).unsqueeze(
                    1
                )
            )
        keys = torch.cat(keys, dim=1).to(self.device)
        querys = torch.cat(querys, dim=1).to(self.device)

        out = self.att(querys, keys, x, mask=None)
        return out


class DecMtanRnn(nn.Module):
    def __init__(
        self,
        input_dim,
        query,
        latent_dim=2,
        nhidden=16,
        embed_time=16,
        num_heads=1,
        linear_hidden_dim=50,
        num_time_emb=3,
        device="cpu",
    ):
        super().__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.num_time_emb = num_time_emb
        self.att = MultiTimeAttention(
            2 * nhidden, 2 * nhidden, embed_time, num_heads, num_time_emb
        )
        self.gru_rnn = nn.GRU(latent_dim, nhidden, bidirectional=True, batch_first=True)
        self.z0_to_obs = nn.Sequential(
            nn.Linear(2 * nhidden, linear_hidden_dim),
            nn.ReLU(),
            nn.Linear(linear_hidden_dim, input_dim),
        )

        periodic = []
        linear_time = []
        for _ in range(self.num_time_emb):
            periodic.append(nn.Linear(1, embed_time - 1))
            linear_time.append(nn.Linear(1, 1))
        self.periodic = nn.ModuleList(periodic)
        self.linear_time = nn.ModuleList(linear_time)

    def learn_time_embedding(self, tt, emb_index):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic[emb_index](tt))
        out1 = self.linear_time[emb_index](tt)
        return torch.cat([out1, out2], -1)

    def forward(self, z, time_steps):
        out, _ = self.gru_rnn(z)
        time_steps = time_steps.cpu()

        keys = []
        querys = []
        for emb_index in range(self.num_time_emb):
            keys.append(
                self.learn_time_embedding(self.query.unsqueeze(0), emb_index).unsqueeze(
                    1
                )
            )
            querys.append(self.learn_time_embedding(time_steps, emb_index).unsqueeze(1))
        keys = torch.cat(keys, dim=1).to(self.device)
        querys = torch.cat(querys, dim=1).to(self.device)

        out = self.att(querys, keys, out)
        out = self.z0_to_obs(out)
        return out


class MegaEncoder(nn.Module):
    def __init__(self, model_conf, data_conf, ref_points):
        super().__init__()
        self.model_conf = model_conf
        self.data_conf = data_conf

        all_emb_size = self.model_conf.features_emb_dim * len(
            self.data_conf.features.embeddings
        )
        all_numeric_size = len(self.data_conf.features.numeric_values)
        self.input_dim = all_emb_size + all_numeric_size

        self.ref_points = ref_points
        self.encoder = EncMtanRnn(
            self.input_dim,
            self.ref_points,
            latent_dim=self.model_conf.latent_dim,
            nhidden=self.model_conf.ref_point_dim,
            embed_time=self.model_conf.time_emb_dim,
            num_heads=self.model_conf.num_heads_enc,
            linear_hidden_dim=self.model_conf.linear_hidden_dim,
            num_time_emb=self.model_conf.num_time_emb,
            device=self.model_conf.device,
        )

    def forward(self, x, time_steps):
        enc_out = self.encoder(x, time_steps)
        return enc_out


class MegaDecoder(nn.Module):
    def __init__(self, model_conf, data_conf, ref_points):
        super().__init__()
        self.model_conf = model_conf
        self.data_conf = data_conf

        all_emb_size = self.model_conf.features_emb_dim * len(
            self.data_conf.features.embeddings
        )
        all_numeric_size = len(self.data_conf.features.numeric_values)
        self.input_dim = all_emb_size + all_numeric_size

        self.ref_points = ref_points
        self.decoder = DecMtanRnn(
            self.input_dim,
            self.ref_points,
            latent_dim=self.model_conf.latent_dim,
            nhidden=self.model_conf.ref_point_dim,
            embed_time=self.model_conf.time_emb_dim,
            num_heads=self.model_conf.num_heads_enc,
            linear_hidden_dim=self.model_conf.linear_hidden_dim,
            num_time_emb=self.model_conf.num_time_emb,
            device=self.model_conf.device,
        )

    def forward(self, z, time_steps):
        out = self.decoder(z, time_steps)

        return out


class MegaNet(nn.Module):
    def __init__(self, data_conf, model_conf):
        super().__init__()
        self.model_conf = model_conf
        self.data_conf = data_conf

        self.ref_points = torch.linspace(0.0, 1.0, self.model_conf.num_ref_points)

        self.preprocessor = FeatureProcessor(
            model_conf=self.model_conf, data_conf=self.data_conf
        )
        self.encoder = MegaEncoder(
            model_conf=model_conf, data_conf=data_conf, ref_points=self.ref_points
        )
        self.decoder = MegaDecoder(
            model_conf=model_conf, data_conf=data_conf, ref_points=self.ref_points
        )

    def forward(self, padded_batch):
        x, time_steps = self.preprocessor(padded_batch)
        enc_out = self.encoder(x, time_steps)

        qz_mean, qz_logstd = torch.split(enc_out, self.model_conf.latent_dim, dim=-1)

        z = sample_z(qz_mean, qz_logstd, self.model_conf.k_iwae)

        iwae_steps = (
            time_steps[None, :, :]
            .repeat(self.model_conf.k_iwae, 1, 1)
            .view(-1, time_steps.shape[1])
        )
        dec_out = self.decoder(z, iwae_steps)
        dec_out = dec_out.view(
            self.model_conf.k_iwae,
            time_steps.shape[0],
            dec_out.shape[1],
            dec_out.shape[2],
        )

        return {
            "x_recon": dec_out,
            "z": z,
            "x": x,
            "time_steps": time_steps,
            "mu": qz_mean,
            "log_std": qz_logstd,
        }

    def loss(self, output, ground_truth):
        """
        output: Dict that is outputed from forward method
        should contain
        1) reconstructed x
        2) initial x
        3) mu from latent
        4) log_std from latent

        ground truth is a Tuple with idx at first pos and label at second
        """

        kl_loss = get_normal_kl(output["mu"], output["log_std"])
        batch_kl_loss = kl_loss.sum([1, 2])

        noise_std_ = torch.zeros(output["x_recon"].size()) + self.model_conf.noise_std
        noise_logvar = torch.log(noise_std_)  # mTAN multiplies by constant 2
        recon_loss = get_normal_nll(output["x"], output["x_recon"], noise_logvar)
        batch_recon_loss = recon_loss.sum([1, 2])

        return {
            "total_loss": batch_recon_loss.mean()
            + self.model_conf.kl_weight * batch_kl_loss.mean(),
            "kl_loss": batch_kl_loss.mean(),
            "recon_loss": batch_recon_loss.mean(),
        }


class Classifier(nn.Module):
    def __init__(self, model_conf, data_conf):
        super().__init__()

        self.model_conf = model_conf
        self.data_conf = data_conf

        self.gru = nn.GRU(
            self.model_conf.latent_dim,
            self.model_conf.classifier_gru_hidden_dim,
            batch_first=True,
        )
        self.net = nn.Sequential(
            nn.Linear(
                self.model_conf.classifier_gru_hidden_dim,
                self.model_conf.classifier_linear_hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                self.model_conf.classifier_linear_hidden_dim,
                self.model_conf.classifier_linear_hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                self.model_conf.classifier_linear_hidden_dim, self.data_conf.num_classes
            ),
        )

    def forward(self, z):
        _, out = self.gru(z)
        return self.net(out.squeeze(0))


class MegaNetSupervised(nn.Module):
    def __init__(self, data_conf, model_conf):
        super().__init__()
        self.model_conf = model_conf
        self.data_conf = data_conf

        self.ref_points = torch.linspace(0.0, 1.0, self.model_conf.num_ref_points).to(
            self.model_conf.device
        )

        self.preprocessor = FeatureProcessor(
            model_conf=self.model_conf, data_conf=self.data_conf
        )
        self.encoder = MegaEncoder(
            model_conf=model_conf, data_conf=data_conf, ref_points=self.ref_points
        )

        self.classifier = Classifier(
            model_conf=self.model_conf, data_conf=self.data_conf
        )

    def forward(self, padded_batch):
        x, time_steps = self.preprocessor(padded_batch)
        enc_out = self.encoder(x, time_steps)

        qz_mean, qz_logstd = torch.split(enc_out, self.model_conf.latent_dim, dim=-1)

        z = sample_z(qz_mean, qz_logstd, self.model_conf.k_iwae)

        # iwae_steps = (
        #     time_steps[None, :, :]
        #     .repeat(self.model_conf.k_iwae, 1, 1)
        #     .view(-1, time_steps.shape[1])
        # )
        # dec_out = self.decoder(z, iwae_steps)
        # dec_out = dec_out.view(
        #     self.model_conf.k_iwae,
        #     time_steps.shape[0],
        #     dec_out.shape[1],
        #     dec_out.shape[2],
        # )

        classifier_out = self.classifier(z)

        return {
            "z": z,
            "x": x,
            "time_steps": time_steps,
            "mu": qz_mean,
            "log_std": qz_logstd,
            "y_pred": classifier_out,
        }

    def loss(self, output, ground_truth):
        """
        output: Dict that is outputed from forward method
        should contain
        1) reconstructed x
        2) initial x
        3) mu from latent
        4) log_std from latent

        ground truth is a Tuple with idx at first pos and label at second
        """

        classification_loss = nn.functional.cross_entropy(
            output["y_pred"], ground_truth[1]
        )

        return {"total_loss": classification_loss}


class MegaNetClassifier(nn.Module):
    def __init__(self, data_conf, model_conf):
        super().__init__()
        self.model_conf = model_conf
        self.data_conf = data_conf

        self.loss_fn = nn.CrossEntropyLoss()
        self.ref_points = torch.linspace(0.0, 1.0, self.model_conf.num_ref_points).to(
            self.model_conf.device
        )

        self.preprocessor = FeatureProcessor(
            model_conf=self.model_conf, data_conf=self.data_conf
        )

        all_emb_size = self.model_conf.features_emb_dim * len(
            self.data_conf.features.embeddings
        )
        all_numeric_size = len(self.data_conf.features.numeric_values)
        self.input_dim = all_emb_size + all_numeric_size

        self.encoder = EncMtanRnnClassification(
            self.input_dim,
            self.ref_points,
            latent_dim=self.model_conf.latent_dim,
            nhidden=self.model_conf.ref_point_dim,
            embed_time=self.model_conf.time_emb_dim,
            num_heads=self.model_conf.num_heads_enc,
            linear_hidden_dim=self.model_conf.linear_hidden_dim,
            num_time_emb=self.model_conf.num_time_emb,
            device=self.model_conf.device,
        )

        self.gru = nn.GRU(
            self.model_conf.ref_point_dim,
            self.model_conf.classifier_gru_hidden_dim,
            batch_first=True,
        )
        self.net = nn.Sequential(
            # nn.BatchNorm1d(self.model_conf.classifier_gru_hidden_dim),
            # nn.LayerNorm(self.model_conf.classifier_gru_hidden_dim),
            nn.Linear(
                self.model_conf.classifier_gru_hidden_dim,
                self.model_conf.classifier_linear_hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                self.model_conf.classifier_linear_hidden_dim,
                self.model_conf.classifier_linear_hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                self.model_conf.classifier_linear_hidden_dim, self.data_conf.num_classes
            ),
        )

    def forward(self, padded_batch):
        x, time_steps = self.preprocessor(padded_batch)
        enc_out = self.encoder(x, time_steps)

        all_hiddens, out = self.gru(enc_out)

        class_out = self.net(out.squeeze(0))
        return {
            "x": x,
            "time_steps": time_steps,
            "y_pred": class_out,
        }

    def loss(self, output, ground_truth):
        """
        output: Dict that is outputed from forward method
        should contain
        1) reconstructed x
        2) initial x
        3) mu from latent
        4) log_std from latent

        ground truth is a Tuple with idx at first pos and label at second
        """

        classification_loss = self.loss_fn(output["y_pred"], ground_truth[1])

        return {"total_loss": classification_loss}


class EmbeddingPredictor(nn.Module):
    def __init__(self, model_conf, data_conf):
        super().__init__()
        self.model_conf = model_conf
        self.data_conf = data_conf

        self.criterion = nn.CrossEntropyLoss(reduction="none")

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
        k_iwae, batch_size, seq_len, out_dim = x_recon.size()

        resized_x = x_recon[:, :, :, : self.categorical_len].view(
            k_iwae * batch_size,
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
            embed_losses[name] = (
                self.criterion(dist.permute(0, 2, 1), padded_batch.payload[name].long())
                .mean(dim=1)
                .mean()
            )

        return embed_losses


class MegaNetCE(nn.Module):
    def __init__(self, data_conf, model_conf):
        super().__init__()
        self.model_conf = model_conf
        self.data_conf = data_conf

        self.register_buffer(
            "ref_points", torch.linspace(0.0, 1.0, self.model_conf.num_ref_points)
        )

        self.preprocessor = FeatureProcessor(
            model_conf=self.model_conf, data_conf=self.data_conf
        )

        self.embedding_predictor = EmbeddingPredictor(
            model_conf=self.model_conf, data_conf=self.data_conf
        )
        self.encoder = MegaEncoder(
            model_conf=model_conf, data_conf=data_conf, ref_points=self.ref_points
        )
        self.decoder = MegaDecoder(
            model_conf=model_conf, data_conf=data_conf, ref_points=self.ref_points
        )

    def forward(self, padded_batch):
        x, time_steps = self.preprocessor(padded_batch)
        enc_out = self.encoder(x, time_steps)

        qz_mean, qz_logstd = torch.split(enc_out, self.model_conf.latent_dim, dim=-1)

        z = sample_z(qz_mean, qz_logstd, self.model_conf.k_iwae)

        iwae_steps = (
            time_steps[None, :, :]
            .repeat(self.model_conf.k_iwae, 1, 1)
            .view(-1, time_steps.shape[1])
        )
        dec_out = self.decoder(z, iwae_steps)
        dec_out = dec_out.view(
            self.model_conf.k_iwae,
            time_steps.shape[0],
            dec_out.shape[1],
            dec_out.shape[2],
        )

        emb_dist = self.embedding_predictor(dec_out)

        return {
            "x_recon": dec_out,
            "z": z,
            "x": x,
            "time_steps": time_steps,
            "mu": qz_mean,
            "log_std": qz_logstd,
            "input_batch": padded_batch,
            "emb_dist": emb_dist,
        }

    def loss(self, output, ground_truth):
        """
        output: Dict that is outputed from forward method
        should contain
        1) reconstructed x
        2) initial x
        3) mu from latent
        4) log_std from latent
        5) padded batch from forward
        6) logits of predicted embeddings
        """

        kl_loss = get_normal_kl(output["mu"], output["log_std"])
        batch_kl_loss = kl_loss.sum([1, 2])

        noise_std_ = torch.zeros(output["x_recon"].size()) + self.model_conf.noise_std
        noise_logvar = torch.log(noise_std_).to(
            self.model_conf.device
        )  # mTAN multiplies by constant 2
        recon_loss = get_normal_nll(output["x"], output["x_recon"], noise_logvar)
        batch_recon_loss = recon_loss.sum([1, 2])

        cross_entropy_losses = self.embedding_predictor.loss(
            output["emb_dist"], output["input_batch"]
        )

        total_ce_loss = torch.sum(
            torch.cat([value.unsqueeze(0) for _, value in cross_entropy_losses.items()])
        )

        losses_dict = {
            "elbo_loss": batch_recon_loss.mean()
            + self.model_conf.kl_weight * batch_kl_loss.mean(),
            "kl_loss": batch_kl_loss.mean(),
            "recon_loss": batch_recon_loss.mean(),
            "total_CE_loss": total_ce_loss,
        }

        total_loss = (
            losses_dict["elbo_loss"] + self.model_conf.CE_weight * total_ce_loss
        )

        losses_dict["total_loss"] = total_loss
        losses_dict.update(cross_entropy_losses)

        return losses_dict
