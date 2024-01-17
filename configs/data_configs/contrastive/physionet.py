from pathlib import Path

import ml_collections


def data_configs():
    config = ml_collections.ConfigDict()

    ########## DATA ##############

    config.train_path = (
        Path(__file__).parent.parent.parent.parent
        / "experiments"
        / "physionet"
        / "data"
        / "train_trx.parquet"
    )

    config.test_path = (
        Path(__file__).parent.parent.parent.parent
        / "experiments"
        / "physionet"
        / "data"
        / "test_trx.parquet"
    )
    config.load_distributed = False
    config.FT_number_objects = [1000, 'all']
    config.post_gen_FT_epochs = 20
    config.track_metric = "roc_auc"

    config.client_list_shuffle_seed = (
        0  # seed for splitting data to train and validation
    )
    config.valid_size = 0.1  # validation size
    config.test_size = 0.0  # pinch_test size
    config.col_id = "user"  # column defining ids. used for sorting data

    features = config.features = ml_collections.ConfigDict()
    # dict below should define all the features that are not numeric with names as keys.
    # "in" parameter is used to clip values at the input.
    features.embeddings = {
        "Gender": {"in": 3, "out": 24, "max_value": 4},
        "ICUType": {"in": 5, "out": 4, "max_value": 5},
        "MechVent": {"in": 2, "out": 4, "max_value": 4},
    }
    # all numeric features are defined here as keys
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

    config.ckpt_path = (
        Path(__file__).parent.parent.parent
        / "experiments"
        / "physionet"
        / "ckpt"
        / "Tr_1l_2h_LN_GR128+LN_2023-09-20_09:32:45"
        / "epoch: 0033 - total_loss: 0.2984 - roc_auc: 0.8421 - loss: 0.2629.ckpt"
    )

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
    train.split_strategy = {
        "split_strategy": "SampleSlices",  # SampleSlices
        "split_count": 5,
        # "seq_len": 50,
        "cnt_min": 15,
        "cnt_max": 150,
    }
    val.split_strategy = {
        "split_strategy": "SampleSlices",  # SampleSlices
        "split_count": 5,
        # "seq_len": 50,
        "cnt_min": 15,
        "cnt_max": 150,
    }
    test.split_strategy = {"split_strategy": "NoSplit"}

    # dropout
    train.dropout = 0.01

    # seq len
    config.use_constant_pad = False
    train.max_seq_len = 200
    val.max_seq_len = 200
    test.max_seq_len = 200

    train.num_workers = 1
    val.num_workers = 1
    test.num_workers = 1

    train.batch_size = 128
    val.batch_size = 128
    test.batch_size = 16

    return config
