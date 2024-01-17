from pathlib import Path

import ml_collections


def data_configs():
    config = ml_collections.ConfigDict()

    ########## DATA ##############

    config.train_path = (
        Path(__file__).parents[3]
        / "experiments"
        / "taobao"
        / "data"
        / "train.parquet"
    )

    config.test_path = (
        Path(__file__).parents[3]
        / "experiments"
        / "taobao"
        / "data"
        / "test.parquet"
    )
    config.load_distributed = False
    config.FT_number_objects = [1000, "all"]
    config.post_gen_FT_epochs = 20

    config.track_metric = "mse"

    config.client_list_shuffle_seed = (
        0  # seed for splitting data to train and validation
    )
    config.valid_size = 0.05  # validation size
    config.test_size = 0.0
    config.col_id = "Index"  # column defining ids. used for sorting data

    features = config.features = ml_collections.ConfigDict()
    # dict below should define all the features that are not numeric with names as keys.
    # "in" parameter is used to clip values at the input.
    features.embeddings = {
        "behavior_type": {"in": 4, "max_value": 5},
        "item_category": {"in": 300, "max_value": 301},
    }
    # all numeric features are defined here as keys
    features.numeric_values = {}

    config.ckpt_path = ()

    # name of target col
    features.target_col = "n_payment_next_7_days"
    config.num_classes = 1

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
    train.max_seq_len = 500
    val.max_seq_len = 500
    test.max_seq_len = 500

    train.num_workers = 1
    val.num_workers = 1
    test.num_workers = 1

    train.batch_size = 128
    val.batch_size = 128
    test.batch_size = 8

    return config
