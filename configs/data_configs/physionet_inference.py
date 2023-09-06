from pathlib import Path

import ml_collections


def data_configs():
    config = ml_collections.ConfigDict()

    ########## DATA ##############

    config.train_path = (
        Path(__file__).parent.parent.parent
        / "experiments"
        / "physionet"
        / "data"
        / "train_trx_supervised.parquet"
    )
    config.test_path = (
        Path(__file__).parent.parent.parent
        / "experiments"
        / "physionet"
        / "data"
        / "test_trx.parquet"
    )

    config.client_list_shuffle_seed = (
        0xAB0BA  # seed for splitting data to train and validation
    )
    config.valid_size = 0.05  # validation size
    config.col_id = "user"  # column defining ids. used for sorting data

    features = config.features = ml_collections.ConfigDict()
    # dict below should define all the features that are not numeric with names as keys.
    # "in" parameter is used to clip values at the input.
    # have not figured out the purpose of "out"
    features.embeddings = {
        "Gender": {"in": 3, "out": 24, "max_value": 4},
        "ICUType": {"in": 5, "out": 4, "max_value": 5},
        "MechVent": {"in": 2, "out": 4, "max_value": 3},
    }
    # all numeric features are defined here as keys
    # seem like its value is technical and is not used anywhere
    features.numeric_values = {
        "Age": None,
        "Height": None,
        "Weight": None,
        "Albumin": None,
        "ALP": None,
        "ALT": None,
        "AST": None,
        "Bilirubin": None,
        "BUN": None,
        "Cholesterol": None,
        "Creatinine": None,
        "DiasABP": None,
        "FiO2": None,
        "GCS": None,
        "Glucose": None,
        "HCO3": None,
        "HCT": None,
        "HR": None,
        "K": None,
        "Lactate": None,
        "Mg": None,
        "MAP": None,
        "Na": None,
        "NIDiasABP": None,
        "NIMAP": None,
        "NISysABP": None,
        "PaCO2": None,
        "PaO2": None,
        "pH": None,
        "Platelets": None,
        "RespRate": None,
        "SaO2": None,
        "SysABP": None,
        "Temp": None,
        "TroponinI": None,
        "TroponinT": None,
        "Urine": None,
        "WBC": None,
    }

    # name of target col
    features.target_col = "target"
    config.num_classes = 2

    ### TIME ###
    config.max_time = 1.0
    config.min_time = 0.0

    # train specific parameters
    train = config.train = ml_collections.ConfigDict()
    # validation specific
    val = config.val = ml_collections.ConfigDict()
    # test params
    test = config.test = ml_collections.ConfigDict()

    # splitters
    train.split_strategy = {"split_strategy": "NoSplit"}
    val.split_strategy = {"split_strategy": "NoSplit"}
    test.split_strategy = {"split_strategy": "NoSplit"}

    # dropout
    train.dropout = 0

    # max seq len
    train.max_seq_len = 200
    val.max_seq_len = 200
    test.max_seq_len = 200

    train.num_workers = 1
    val.num_workers = 1
    test.num_workers = 1

    train.batch_size = 16
    val.batch_size = 16
    test.batch_size = 16

    ### Path to trained model ###
    config.ckpt_path = (
        Path(__file__).parent.parent.parent
        / "experiments"
        / "physionet"
        / "experiments"
        / "physionet"
        / "ckpt"
        / "noVAE_2023-09-06_08:04:54"
        / "epoch: 0010 - total_loss: 0.2825 - kl_loss: 316.8 - recon_loss: 1.112e+10 - classification_loss: 0.2825 - loss: 0.3407.ckpt"
    )

    config.train_embed_path = (
        Path(__file__).parent.parent.parent
        / "experiments"
        / "physionet"
        / "data"
        / "train_embed.csv"
    )
    config.valid_embed_path = (
        Path(__file__).parent.parent.parent
        / "experiments"
        / "physionet"
        / "data"
        / "valid_embed.csv"
    )
    config.test_embed_path = (
        Path(__file__).parent.parent.parent
        / "experiments"
        / "physionet"
        / "data"
        / "test_embed.csv"
    )

    return config
