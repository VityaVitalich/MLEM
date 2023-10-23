from pathlib import Path

import ml_collections


def data_configs():
    config = ml_collections.ConfigDict()

    ########## DATA ##############

    config.train_path = (
        Path(__file__).parent.parent.parent
        / "experiments"
        / "sber"
        / "data"
        / "train.parquet"
    )
    config.train_supervised_path = (
        Path(__file__).parent.parent.parent
        / "experiments"
        / "sber"
        / "data"
        / "train_supervised.parquet"
    )

    config.test_path = (
        Path(__file__).parent.parent.parent
        / "experiments"
        / "sber"
        / "data"
        / "test.parquet"
    )

    config.client_list_shuffle_seed = (
        0  # 0xAB0BA  # seed for splitting data to train and validation
    )
    config.valid_size = 0.1  # validation size
    config.col_id = "user"  # column defining ids. used for sorting data

    features = config.features = ml_collections.ConfigDict()
    # dict below should define all the features that are not numeric with names as keys.
    # "in" parameter is used to clip values at the input.
    # have not figured out the purpose of "out"
    features.embeddings = {
        "trx_country":   {"in": 1000, "out": 24, "max_value": 1 + 2},
        "mcc_code":      {"in": 1000, "out": 24, "max_value": 1 + 300},
        "trans_type":    {"in": 1000, "out": 24, "max_value": 1 + 300},
        "card_cat_cd":   {"in": 1000, "out": 24, "max_value": 1 + 3},
        "ipt_name":      {"in": 1000, "out": 24, "max_value": 1 + 7},
        "iso_crncy_cd":  {"in": 1000, "out": 24, "max_value": 1 + 2},
        "ecom_fl":       {"in": 1000, "out": 24, "max_value": 1 + 2},
        "trx_direction": {"in": 1000, "out": 24, "max_value": 1 + 2},
    }
    # all numeric features are defined here as keys
    # seem like its value is technical and is not used anywhere
    features.numeric_values = {
        "amt": None,
    }

    config.ckpt_path = (
        Path(__file__).parent.parent.parent
        / "experiments"
        / "sber"
        / "ckpt"
    )

    # name of target col
    features.target_col = "pl"
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
        "split_strategy": "SampleUniform",  # SampleSlices
        "split_count": 4,
        "seq_len": 50,
        # "cnt_min": 15,
        # "cnt_max": 150,
    }
    val.split_strategy = {
        "split_strategy": "SampleUniform",  # SampleSlices
        "split_count": 4,
        "seq_len": 50,
        # "cnt_min": 15,
        # "cnt_max": 150,
    }
    test.split_strategy = {"split_strategy": "NoSplit"}

    # dropout
    train.dropout = 0.05

    # seq len
    train.max_seq_len = 200
    val.max_seq_len = 200
    test.max_seq_len = 200

    train.num_workers = 1
    val.num_workers = 1
    test.num_workers = 1

    train.batch_size = 128
    val.batch_size = 128
    test.batch_size = 128

    return config
