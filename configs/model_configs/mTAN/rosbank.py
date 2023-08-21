import ml_collections


def model_configs():
    config = ml_collections.ConfigDict()

    ########## Embeddings ##############

    config.features_emb_dim = 16
    config.num_ref_points = 64
    config.latent_dim = 2
    config.ref_point_dim = 16
    config.time_emb_dim = 16
    config.num_heads_enc = 2
    config.linear_hidden_dim = 32
    config.k_iwae = 1
    config.device = "cpu"

    return config
