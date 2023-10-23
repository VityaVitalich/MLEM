import ml_collections


def model_configs():
    config = ml_collections.ConfigDict()

    config.model_name = "GRUClassifier"
    config.predict_head = "Linear"  # Linear or Identity

    # Vitya NIPS
    config.batch_first_encoder = [True, False]

    ### EMBEDDINGS ###
    # features_emb_dim is dimension of nn.Embedding applied to categorical features
    config.features_emb_dim = [4, 8, 16, 32]
    config.use_numeric_emb = False
    config.numeric_emb_size = 8
    ### RNN + LINEAR ###
    config.classifier_gru_hidden_dim = [16, 23, 64, 128]
    config.classifier_linear_hidden_dim = 300

    ### TRANSFORMER ###
    config.encoder = ["Identity", "TransformerEncoder"]  # Identity or TransformerEncoder 
    config.num_enc_layers = [1, 2]
    config.num_heads_enc = [1, 2, 4]

    ### NORMALIZATIONS ###
    config.pre_gru_norm = "Identity"
    config.post_gru_norm = "LayerNorm"
    config.encoder_norm = ["Identity", "LayerNorm"] # if TransformerEncoder -> LayerNorm. else Identity

    ### DROPOUT ###
    config.after_enc_dropout = [0.0, 0.1, 0.2]

    # we use GRU, not applicable
    # ### CONVOLUTIONAL ###
    # conv = config.conv = ml_collections.ConfigDict()
    # conv.out_channels = 16
    # conv.kernels = [3, 5, 9]
    # conv.dilations = [3, 5, 9]
    # conv.num_stacks = 2
    # conv.proj = "Linear"

    ### ACTIVATION ###
    config.activation = ["ReLU", "LeakyReLU", "Mish", "Tanh"]

    ### TIME TRICKS ###
    config.num_time_blocks = 50 #[4, 16] 
    config.time_preproc = (
        "Identity"  # Identity or TimeConcater or MultiTimeSummator 
    )
    config.entropy_weight = 0.0

    ### LOSS ###
    loss = config.loss = ml_collections.ConfigDict()
    loss.sampling_strategy = "HardNegativePair"
    loss.loss_fn = ["ContrastiveLoss", "InfoNCELoss", "DecoupledInfoNCELoss", "DecoupledPairwiseInfoNCELoss", "RINCELoss"]
    loss.margin = [0.0, 0.1, 0.3, 0.5, 1.0]  # ContrastiveLoss only
    loss.neg_count = 5
    loss.projector = ["Identity", "Linear", "MLP"] # all losses
    loss.project_dim = [32, 64, 128, 256]  # all losses
    loss.temperature = [0.01, 0.03, 0.1, 0.3, 1.0]  # all except ContrastiveLoss
    loss.angular_margin = [0.0, 0.3, 0.5, 0.7]  # InfoNCELoss only
    loss.q = [0.01, 0.03, 0.1, 0.3]  # RINCELoss only
    loss.lam = [0.003, 0.01, 0.03, 0.1, 0.3]  # RINCELoss only

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
    # config.k_iwae = 1
    # # noise to diagonal matrix of output distribution
    # config.noise_std = 0.01
    # # weight of kl term in loss
    # config.kl_weight = 0.0
    # config.CE_weight = 0
    # config.reconstruction_weight = 0
    # config.classification_weight = 1

    ### DEVICE + OPTIMIZER ###
    config.device = "cuda"

    config.lr = [3e-4, 1e-3, 3e-3]
    config.weight_decay = [1e-5, 1e-4, 1e-3]
    config.cv_splits = 5

    config.comments = ""
    return config