from pathlib import Path

import ml_collections


def data_configs():
    config = ml_collections.ConfigDict()

    ########## DATA ##############

    config.train_path = (
        Path(__file__).parent.parent.parent
        / "experiments"
        / "half_moon"
        / "data"
        / "train_trx.parquet"
    )

    config.client_list_shuffle_seed = (
        0  # seed for splitting data to train and validation
    )
    config.valid_size = 0.1  # validation size
    config.col_id = "col_id"  # column defining ids. used for sorting data

    features = config.features = ml_collections.ConfigDict()
    # dict below should define all the features that are not numeric with names as keys.
    # "in" parameter is used to clip values at the input.
    features.embeddings = {}
    # all numeric features are defined here as keys
    features.numeric_values = {"0": "identity", "1": "identity"}

    # name of target col
    features.target_col = "target"
    config.num_classes = 2

    ### TIME ###
    config.max_time = 1
    config.min_time = 0

    # train specific parameters
    train = config.train = ml_collections.ConfigDict()
    # validation specific
    val = config.val = ml_collections.ConfigDict()

    # splitters
    # train.split_strategy = {
    #     "split_strategy": "SampleSlices",
    #     "split_count": 5,
    #     "cnt_min": 15,
    #     "cnt_max": 150,
    # }
    # val.split_strategy = {
    #     "split_strategy": "SampleSlices",
    #     "split_count": 5,
    #     "cnt_min": 15,
    #     "cnt_max": 150,
    # }

    train.split_strategy = {"split_strategy": "NoSplit"}
    val.split_strategy = {"split_strategy": "NoSplit"}

    # dropout
    train.dropout = 0.0
    train.max_seq_len = 2

    val.max_seq_len = 2

    train.num_workers = 1
    val.num_workers = 1

    train.batch_size = 64
    val.batch_size = 64

    return config
