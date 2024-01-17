import ml_collections


def model_configs():
    config = ml_collections.ConfigDict()

    config.model_name = "GRUClassifier"
    config.predict_head = "Identity"  # Linear or Identity

    config.batch_first_encoder = False

    ### EMBEDDINGS ###
    # features_emb_dim is dimension of nn.Embedding applied to categorical features
    config.features_emb_dim = 8
    config.use_numeric_emb = False
    config.numeric_emb_size = 8

    ### RNN + LINEAR ###
    config.classifier_gru_hidden_dim = 64
    config.classifier_linear_hidden_dim = 300  # Used only in MTAN

    ### TRANSFORMER ###
    config.encoder = "Identity"  # Identity or TransformerEncoder
    config.num_enc_layers = 1
    config.num_heads_enc = 1

    ### NORMALIZATIONS ###
    config.pre_gru_norm = "Identity"
    config.post_gru_norm = "LayerNorm"
    config.encoder_norm = (
        "Identity"  # if TransformerEncoder -> LayerNorm. else Identity
    )

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
    config.num_time_blocks = 50  # [4, 16]
    config.time_preproc = "Identity"  # Identity or TimeConcater or MultiTimeSummator
    config.entropy_weight = 0.0

    ### LOSS ###
    loss = config.loss = ml_collections.ConfigDict()
    loss.sampling_strategy = "HardNegativePair"
    loss.neg_count = 5
    loss.loss_fn = "CrossEntropy"  # "ContrastiveLoss" or CrossEntropy
    loss.margin = 0.5

    ### MTAND ###
    # # number of reference points on encoder
    # config.num_ref_points = 128
    # # latent dimension for mu and sigma
    # config.latent_dim = 2
    # # dimension of reference points after mTAN layer
    # # in fact is the dimension of output linear in attention
    # config.ref_point_dim = 128
    # # dim of each time emb
    # config.time_emb_dim = 16
    # # number of heads in mTAN attention
    # config.num_heads_enc = 2
    # # dim in FF layer after attention
    # config.linear_hidden_dim = 50
    # # number of time embeddings
    # config.num_time_emb = 3

    ### VAE PARAMS ###
    # number of iwae samples
    config.k_iwae = 1
    # noise to diagonal matrix of output distribution
    config.noise_std = 0.01
    # weight of kl term in loss
    config.kl_weight = 0.0
    config.CE_weight = 0
    config.reconstruction_weight = 0
    config.classification_weight = 1

    ### DEVICE + OPTIMIZER ###
    config.device = "cuda"

    config.lr = 3e-3
    config.weight_decay = 1e-3
    config.cv_splits = 5

    config.comments = ""
    return config
