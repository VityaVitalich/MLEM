import ml_collections


def model_configs():
    config = ml_collections.ConfigDict()

    config.model_name = "GRUClassifier"
    config.predict_head = "Linear"  # Linear or Identity

    ### EMBEDDINGS ###
    # features_emb_dim is dimension of nn.Embedding applied to categorical features
    config.features_emb_dim = 8
    config.use_numeric_emb = False
    config.numeric_emb_size = 8

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
