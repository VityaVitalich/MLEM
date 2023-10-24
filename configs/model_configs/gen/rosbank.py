import ml_collections


def model_configs():
    config = ml_collections.ConfigDict()

    config.model_name = "SeqGen"
    config.predict_head = "Linear"  # Linear or Identity

    # Vitya NIPS
    config.preENC_TR = False
    config.batch_first_encoder = True

    ### EMBEDDINGS ###
    # features_emb_dim is dimension of nn.Embedding applied to categorical features
    config.features_emb_dim = 12
    config.use_numeric_emb = True
    config.numeric_emb_size = 12
    config.encoder_feature_mixer = True
    config.decoder_feature_mixer = True

    ### ENCODER ###
    config.encoder = "GRU"
    config.encoder_hidden = 128
    config.encoder_num_layers = 1

    ### TRANSFORMER ENCODER ###
    config.encoder_num_heads = 1

    ### DECODER ###
    config.decoder = "GRU"
    config.decoder_hidden = 32
    config.decoder_num_layers = 1

    ### TRANSFORMER DECODER ###
    config.decoder_heads = 1

    ### NORMALIZATIONS ###
    config.pre_encoder_norm = "Identity"
    config.post_encoder_norm = "Identity"
    config.decoder_norm = "Identity"
    config.encoder_norm = "Identity"

    ### GENERATED EMBEDDINGS LOSS ###
    config.generative_embeddings_loss = False
    config.gen_emb_loss_type = "l1"

    ### DROPOUT ###
    config.after_enc_dropout = 0.05

    ### ACTIVATION ###
    config.activation = "ReLU"

    ### TIME ###
    config.use_deltas = True
    config.delta_weight = 5

    ### DISCRIMINATOR ###
    config.use_discriminator = True

    ### LOSS ###
    config.mse_weight = 1
    config.CE_weight = 1
    config.l1_weight = 0.001
    config.gen_emb_weight = 1
    config.D_weight = 1

    ### DEVICE + OPTIMIZER ###
    config.device = "cuda:1"

    config.lr = 3e-3
    config.weight_decay = 1e-3
    config.cv_splits = 5

    config.comments = ""
    return config
