import ml_collections


def model_configs():
    config = ml_collections.ConfigDict()

    ########## Embeddings ##############

    # features_emb_dim is dimension of nn.Embedding applied to categorical features
    config.features_emb_dim = 8
    # number of reference points on encoder
    config.num_ref_points = 64
    # latent dimension for mu and sigma
    config.latent_dim = 2
    # dimension of reference points after mTAN layer
    # in fact is the dimension of output linear in attention
    config.ref_point_dim = 32
    # dim of each time emb
    config.time_emb_dim = 16
    # number of heads in mTAN attention
    config.num_heads_enc = 2
    # dim in FF layer after attention
    config.linear_hidden_dim = 16
    # number of time embeddings
    config.num_time_emb = 1
    # number of hidden gru
    config.classifier_gru_hidden_dim = 16
    config.classifier_linear_hidden_dim = 300

    # number of iwae samples
    config.k_iwae = 1
    # noise to diagonal matrix of output distribution
    config.noise_std = 0.01
    # weight of kl term in loss
    config.kl_weight = 0.0
    config.CE_weight = 0
    config.reconstruction_weight = 0
    config.classification_weight = 1
    config.device = "cuda"

    return config
