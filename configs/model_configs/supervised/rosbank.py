import ml_collections


def model_configs():
    config = ml_collections.ConfigDict()

    config.model_name = "GRUClassifier"
    config.predict_head = "Linear"  # Linear or Identity

    # Vitya NIPS
    config.preENC_TR = False
    config.batch_first_encoder = True

    ### EMBEDDINGS ###
    # features_emb_dim is dimension of nn.Embedding applied to categorical features
    config.features_emb_dim = 32
    config.use_numeric_emb = True
    config.numeric_emb_size = 32
    config.encoder_feature_mixer = False

    ### ENCODER ###
    config.encoder = "GRU"  # GRU LSTM TR
    config.encoder_hidden = 512
    config.encoder_num_layers = 1

    ### TRANSFORMER ENCODER ###
    config.encoder_num_heads = 1
    config.encoder_dim_ff = 256

    ### TIME DELTA ###
    config.use_deltas = True
    config.time_embedding = 0

    ### NORMALIZATIONS ###
    config.pre_encoder_norm = "Identity"
    config.post_encoder_norm = "LayerNorm"
    config.encoder_norm = "Identity"
    # if TransformerEncoder -> LayerNorm. else Identity. TODO check this!!!

    ### DROPOUT ###
    config.after_enc_dropout = 0.03

    ### ACTIVATION ###
    config.activation = "LeakyReLU"

    ### TIME TRICKS ###
    config.num_time_blocks = 50  # [4, 16]
    config.time_preproc = "Identity"  # Identity or TimeConcater or MultiTimeSummator
    config.entropy_weight = 0.0

    ### LOSS ###
    loss = config.loss = ml_collections.ConfigDict()
    loss.sampling_strategy = "HardNegativePair"
    loss.loss_fn = "CrossEntropy"
    loss.margin = 0.5  # ContrastiveLoss only
    loss.neg_count = 5
    loss.projector = "Identity"  # all losses
    loss.project_dim = 32  # all losses
    loss.temperature = 0.1  # all except ContrastiveLoss
    loss.angular_margin = 0.3  # InfoNCELoss only
    loss.q = 0.03  # RINCELoss only
    loss.lam = 0.01  # RINCELoss only

    ### DEVICE + OPTIMIZER ###
    config.device = "cuda"

    config.lr = 0.001
    config.weight_decay = 0.0

    config.comments = ""
    return config

    ### CKCONV ###
    ckconv = config.ckconv = ml_collections.ConfigDict()
    ckconv.hidden_channels = 64
    ckconv.num_blocks = 3
    ckconv.kernel_hidden_channels = 64
    ckconv.kernel_activation = "Sine"
    ckconv.kernel_norm = ""
    ckconv.omega = 30
    ckconv.dropout = 0.1
    ckconv.weight_dropout = 0.0
    ckconv.use_real_time = True
    return config
