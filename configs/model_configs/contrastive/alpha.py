import ml_collections


def model_configs():
    config = ml_collections.ConfigDict()

    config.model_name = "GRUClassifier"
    config.predict_head = "Identity"  # Linear or Identity

    # Vitya NIPS
    config.batch_first_encoder = True

    ### EMBEDDINGS ###
    # features_emb_dim is dimension of nn.Embedding applied to categorical features
    config.features_emb_dim = 16
    config.use_numeric_emb = False
    config.numeric_emb_size = 8
    config.encoder_feature_mixer = False
    config.time_embedding = 2

    ### RNN + LINEAR ###
    config.classifier_gru_hidden_dim = 800

    ### TIME DELTA ###
    config.use_deltas = False
    ### TRANSFORMER ###
    config.encoder = "Identity"  # Identity or TransformerEncoder
    config.num_enc_layers = 1
    config.num_heads_enc = 1

    ### NORMALIZATIONS ###
    config.pre_gru_norm = "Identity"
    config.post_gru_norm = "Identity"
    config.encoder_norm = "Identity"
    # if TransformerEncoder -> LayerNorm. else Identity. TODO check this!!!

    ### DROPOUT ###
    config.after_enc_dropout = 0.0

    ### ACTIVATION ###
    config.activation = "LeakyReLU"

    ### TIME TRICKS ###
    config.num_time_blocks = 50  # [4, 16]
    config.time_preproc = "Identity"  # Identity or TimeConcater or MultiTimeSummator
    config.entropy_weight = 0.0

    ### LOSS ###
    loss = config.loss = ml_collections.ConfigDict()
    loss.sampling_strategy = "HardNegativePair"
    loss.loss_fn = "ContrastiveLoss"
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
