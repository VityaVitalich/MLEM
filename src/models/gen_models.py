import torch
import torch.nn as nn
from . import preprocessors as prp
from ..trainers.losses import get_loss


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


class GRUClassifier(BaseMixin):
    def __init__(self, model_conf, data_conf):
        super().__init__(model_conf, data_conf)

        self.gru = nn.GRU(
            self.input_dim,
            self.model_conf.classifier_gru_hidden_dim,
            batch_first=True,
        )

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

        y = self.out_linear(last_hidden)

        return y


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
        all_numeric_size = len(self.data_conf.features.numeric_values)
        self.input_dim = all_emb_size + all_numeric_size

        self.encoder = nn.GRU(
            self.input_dim,
            self.model_conf.encoder_hidden,
            batch_first=True,
        )
        self.decoder = nn.Linear(self.model_conf.encoder_hidden, self.input_dim)

        self.embedding_predictor = EmbeddingPredictor(
            model_conf=self.model_conf, data_conf=self.data_conf
        )

    def forward(self, padded_batch):
        x, time_steps = self.preprocessor(padded_batch)
        all_hid, hn = self.encoder(x)

        pred = self.decoder(all_hid)[:, :-1, :]

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
