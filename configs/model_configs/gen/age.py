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
    config.encoder_feature_mixer = False
    config.decoder_feature_mixer = False

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
    config.gen_emb_loss_type = "cosine"

    ### DROPOUT ###
    config.after_enc_dropout = 0.05

    ### ACTIVATION ###
    config.activation = "ReLU"

    ### TIME ###
    config.use_deltas = True
    config.time_embedding = 0
    config.use_log_delta = False
    config.delta_weight = 10

    ### LOSS ###
    config.mse_weight = 1
    config.CE_weight = 1
    config.l1_weight = 0.0001
    config.gen_emb_weight = 10
    config.D_weight = 20

    ### DEVICE + OPTIMIZER ###
    config.device = "cuda"

    config.lr = 3e-3
    config.weight_decay = 1e-3
    config.cv_splits = 5

    config.use_discriminator = False
    config.comments = ""
    config.gen_len = 500
    config.genval = genval_config()
    config.D = d_config()
    return config


def genval_config():
    config = ml_collections.ConfigDict()

    config.model_name = "GRUClassifier"
    config.predict_head = "Linear"  # Linear or Identity

    # Vitya NIPS
    config.batch_first_encoder = False

    ### EMBEDDINGS ###
    # features_emb_dim is dimension of nn.Embedding applied to categorical features
    config.features_emb_dim = 8
    config.use_numeric_emb = True
    config.numeric_emb_size = 8
    config.encoder_feature_mixer = False

    ### TIME DELTA ###
    config.use_deltas = True
    config.time_embedding = 2

    ### RNN + LINEAR ###
    config.classifier_gru_hidden_dim = 64

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

    ### CONVOLUTIONAL ###
    conv = config.conv = ml_collections.ConfigDict()
    conv.out_channels = 32
    conv.kernels = [3, 5, 9]
    conv.dilations = [3, 5, 9]
    conv.num_stacks = 3
    conv.proj = "Linear"

    ### ACTIVATION ###
    config.activation = "ReLU"

    ### TIME TRICKS ###
    config.num_time_blocks = [
        1,
        8,
        16,
        32,
        64,
    ]
    config.time_preproc = "Identity"  # Identity or TimeConcater or MultiTimeSummator
    config.entropy_weight = 0.0

    ### LOSS ###
    loss = config.loss = ml_collections.ConfigDict()
    loss.sampling_strategy = "HardNegativePair"
    loss.neg_count = 5
    loss.loss_fn = "CrossEntropy"  # "ContrastiveLoss" or CrossEntropy
    loss.margin = 0.5

    ### DEVICE + OPTIMIZER ###
    config.device = "cuda"

    config.lr = 3e-3
    config.weight_decay = 1e-3
    config.cv_splits = 5

    config.comments = ""
    return config


def d_config():
    config = ml_collections.ConfigDict()

    config.model_name = "GRUClassifier"
    config.predict_head = "Linear"  # Linear or Identity

    # Vitya NIPS
    config.batch_first_encoder = False

    ### EMBEDDINGS ###
    # features_emb_dim is dimension of nn.Embedding applied to categorical features
    config.features_emb_dim = 8
    config.use_numeric_emb = True
    config.numeric_emb_size = 8
    config.encoder_feature_mixer = False

    ### TIME DELTA ###
    config.use_deltas = True
    config.time_embedding = 2

    ### RNN + LINEAR ###
    config.classifier_gru_hidden_dim = 64
    config.classifier_linear_hidden_dim = 300  # Used only in MTAN

    ### TRANSFORMER ###
    config.encoder = "TransformerEncoder"  # IDnetity or TransformerEncoder
    config.num_enc_layers = 1
    config.num_heads_enc = 1

    ### NORMALIZATIONS ###
    config.pre_gru_norm = "Identity"
    config.post_gru_norm = "LayerNorm"
    config.encoder_norm = "LayerNorm"

    ### DROPOUT ###
    config.after_enc_dropout = 0.3

    ### CONVOLUTIONAL ###
    conv = config.conv = ml_collections.ConfigDict()
    conv.out_channels = 32
    conv.kernels = [3, 5, 9]
    conv.dilations = [3, 5, 9]
    conv.num_stacks = 3
    conv.proj = "Linear"

    ### ACTIVATION ###
    config.activation = "ReLU"

    ### TIME TRICKS ###
    config.num_time_blocks = 50
    config.time_preproc = "Identity"  # Identity or TimeConcater or MultiTimeSummator
    config.entropy_weight = 0.0

    ### LOSS ###
    loss = config.loss = ml_collections.ConfigDict()
    loss.sampling_strategy = "HardNegativePair"
    loss.neg_count = 5
    loss.loss_fn = "CrossEntropy"  # "ContrastiveLoss" or CrossEntropy
    loss.margin = 0.5

    ### DEVICE + OPTIMIZER ###
    config.device = "cuda"

    config.lr = 3e-3
    config.weight_decay = 1e-3

    config.comments = ""
    return config
