def model_conf_sugg(model_conf, trial, data):
    name, value_range, value_type = data
    if value_type is str:
        model_conf[name] = trial.suggest_categorical(name, value_range)
    else:
        assert len(value_range) == 2
    if value_type is int:    
        model_conf[name] = trial.suggest_int(name, value_range[0], value_range[1])
    elif value_type is float:
        model_conf[name] = trial.suggest_float(name, value_range[0], value_range[1])
    elif value_type == "int_log":
        model_conf[name] = trial.suggest_int(name, value_range[0], value_range[1], log=True)
    elif value_type == "float_log":
        model_conf[name] = trial.suggest_float(name, value_range[0], value_range[1], log=True)
    else:
        raise NotImplementedError
    return model_conf

def contrastive_base(trial, model_conf, data_conf):
    # Looking for max gpu usage, debuging alongside
    for param in (
        ("batch_first_encoder", [True, False], str),
        ("features_emb_dim", [4, 32], "int_log"),
        ("use_numeric_emb", [True, False], str),
        # ("numeric_emb_size", [4, 32], "int_log"),
        ("encoder_feature_mixer", [False], str),
        ("classifier_gru_hidden_dim", [16, 1024], "int_log"),
        ("use_deltas", [True, False], str),
        ("encoder", ["Identity", "TransformerEncoder"], str),
        ("num_enc_layers", [1, 2], int),
        ("num_heads_enc", [1, 1], int), # TODO complicated not to fail
        ("after_enc_dropout", [0.0, 0.4], float),
        ("activation", ["ReLU", "LeakyReLU", "Mish", "Tanh"], str),
        ("lr", [3e-4, 3e-3], "float_log"),
        ("weight_decay", [0, 1e-3], "float_log"),
    ):
        model_conf_sugg(model_conf, trial, param)
    # model_conf["features_emb_dim"] = model_conf["features_emb_dim"] // 2 * 2 # make sure that features_emb_dim is even
    model_conf["numeric_emb_size"] = model_conf["features_emb_dim"]
    if model_conf["encoder"] == "TransformerEncoder":
        model_conf_sugg(model_conf, trial, ("encoder_norm", ["Identity", "LayerNorm"], str))  # important to know corr to encoder
    # if model_conf["encoder_feature_mixer"]:
    #     model_conf["time_embedding"] = model_conf["features_emb_dim"] // 2
    elif model_conf["encoder"] == "Identity":
        model_conf["encoder_norm"] = "Identity"

    # for param in (
    #     ("loss_fn", ["ContrastiveLoss"], str),  # , "InfoNCELoss", "DecoupledInfoNCELoss", "RINCELoss", "DecoupledPairwiseInfoNCELoss"]),
    #     ("projector", ["Identity", "Linear", "MLP"], str),
    #     ("project_dim", [32, 256], "int_log"),
    # ):
    #     model_conf_sugg(model_conf.loss, trial, param)
    # if model_conf.loss.loss_fn == "ContrastiveLoss":
    #     model_conf.loss.margin = trial.suggest_float("margin", 0.0, 1.0)
    # else:
    #     model_conf.loss.temperature = trial.suggest_float("temperature", 0.01, 1.0, log=True)
    # if model_conf.loss.loss_fn == "InfoNCELoss":
    #     model_conf.loss.angular_margin = trial.suggest_float("angular_margin", 0.0, 0.7)
    # elif model_conf.loss.loss_fn == "RINCELoss":
    #     model_conf.loss.q = trial.suggest_float(
    #         "q", 0.01, 0.3, log=True
    #     )
    #     model_conf.loss.lam = trial.suggest_float(
    #         "lam", 0.003, 0.3, log=True
    #     )
    return trial, model_conf, data_conf

def contrasive_loss_27_11_23(trial, model_conf, data_conf):
    trial, model_conf, data_conf = contrastive_base(trial, model_conf, data_conf)
    for param in (
        ("loss_fn", ["ContrastiveLoss"], str), 
        ("projector", ["Identity", "Linear", "MLP"], str),
        ("project_dim", [32, 256], "int_log"),
    ):
        model_conf_sugg(model_conf.loss, trial, param)
    model_conf.loss.margin = trial.suggest_float("margin", 0.0, 1.0)
    return trial, model_conf, data_conf

def decoupledinfoloss_27_11_23(trial, model_conf, data_conf):
    trial, model_conf, data_conf = contrastive_base(trial, model_conf, data_conf)
    for param in (
        ("loss_fn", ["DecoupledInfoNCELoss"], str), 
        ("projector", ["Identity", "Linear", "MLP"], str),
        ("project_dim", [32, 256], "int_log"),
    ):
        model_conf_sugg(model_conf.loss, trial, param)
    model_conf.loss.temperature = trial.suggest_float("temperature", 0.01, 1.0, log=True)
    return trial, model_conf, data_conf

