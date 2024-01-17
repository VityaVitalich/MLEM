from pathlib import Path

import ml_collections


def data_configs():
    config = ml_collections.ConfigDict()

    ########## DATA ##############

    config.train_path = (
        Path(__file__).parent.parent.parent.parent
        / "experiments"
        / "age"
        / "data"
        / "train_trx.parquet"
    )

    config.test_path = (
        Path(__file__).parent.parent.parent.parent
        / "experiments"
        / "age"
        / "data"
        / "test_trx.parquet"
    )
    config.load_distributed = False
    config.FT_number_objects = [1000, "all"]
    config.post_gen_FT_epochs = 20

    config.track_metric = "accuracy"

    config.client_list_shuffle_seed = (
        0   # seed for splitting data to train and validation
    )
    config.valid_size = 0.05  # validation size
    config.test_size = 0.0
    config.col_id = "client_id"  # column defining ids. used for sorting data

    features = config.features = ml_collections.ConfigDict()
    # dict below should define all the features that are not numeric with names as keys.
    # "in" parameter is used to clip values at the input.
    features.embeddings = {
        "small_group": {"in": 250, "out": 250, "max_value": 252},
    }
    # all numeric features are defined here as keys
    features.numeric_values = {
        "amount_rur": "Identity",
    }

    # name of target col
    features.target_col = "target"
    config.num_classes = 4

    ### TIME ###
    config.max_time = 729.0
    config.min_time = 0.0

    # train specific parameters
    train = config.train = ml_collections.ConfigDict()
    # validation specific
    val = config.val = ml_collections.ConfigDict()
    # test params
    test = config.test = ml_collections.ConfigDict()

    # splitters
    train.split_strategy = {
        "split_strategy": "SampleSlices",
        "split_count": 5,
        "cnt_min": 25,
        "cnt_max": 200,
    }
    val.split_strategy = {
        "split_strategy": "SampleSlices",
        "split_count": 5,
        "cnt_min": 25,
        "cnt_max": 100,
    }
    test.split_strategy = {"split_strategy": "NoSplit"}

    # dropout
    train.dropout = 0.01

    # seq len
    config.min_seq_len = 25
    config.use_constant_pad = False
    train.max_seq_len = 1200
    val.max_seq_len = 1200
    test.max_seq_len = 1200

    train.num_workers = 1
    val.num_workers = 1
    test.num_workers = 1

    train.batch_size = 64
    val.batch_size = 64
    test.batch_size = 8

    return config
