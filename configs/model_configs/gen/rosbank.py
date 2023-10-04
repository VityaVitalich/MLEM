import ml_collections


def model_configs():
    config = ml_collections.ConfigDict()

    config.model_name = "GRUGen"
    config.predict_head = "Linear"  # Linear or Identity

    ### EMBEDDINGS ###
    # features_emb_dim is dimension of nn.Embedding applied to categorical features
    config.features_emb_dim = 8

    ### ENCODER ###
    config.encoder_hidden = 64

    ### DECODER ###
    config.decoder_gru_hidden = 64
    config.decoder_num_layers = 1

    ### TRANSFORMER ###
    config.encoder = "Identity"  # IDnetity or TransformerEncoder
    config.num_enc_layers = 1
    config.num_heads_enc = 1

    ### NORMALIZATIONS ###
    config.pre_gru_norm = "Identity"
    config.post_gru_norm = "LayerNorm"
    config.encoder_norm = "Identity"

    ### DROPOUT ###
    config.after_enc_dropout = 0.0

    ### ACTIVATION ###
    config.activation = "ReLU"

    ### LOSS ###
    config.mse_weight = 1
    config.CE_weight = 1
    config.delta_weight = 10000

    ### DEVICE + OPTIMIZER ###
    config.device = "cuda"

    config.lr = 3e-3
    config.weight_decay = 1e-3
    config.cv_splits = 5

    config.comments = ""
    return config
