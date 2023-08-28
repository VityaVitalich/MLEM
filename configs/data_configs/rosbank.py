from pathlib import Path

import ml_collections


def data_configs():
    config = ml_collections.ConfigDict()

    ########## DATA ##############

    config.train_path = (
        Path(__file__).parent.parent.parent
        / "experiments"
        / "rosbank"
        / "data"
        / "train_trx.parquet"
    )

    config.client_list_shuffle_seed = (
        0xAB0BA  # seed for splitting data to train and validation
    )
    config.valid_size = 0.05  # validation size
    config.col_id = "cl_id"  # column defining ids. used for sorting data

    features = config.features = ml_collections.ConfigDict()
    # dict below should define all the features that are not numeric with names as keys.
    # "in" parameter is used to clip values at the input.
    # have not figured out the purpose of "out"
    features.embeddings = {
        "mcc": {"in": 100, "out": 24, "max_value": 400},
        "channel_type": {"in": 4, "out": 4, "max_value": 400},
        "currency": {"in": 4, "out": 4, "max_value": 400},
        "trx_category": {"in": 10, "out": 4, "max_value": 400},
    }
    # all numeric features are defined here as keys
    # seem like its value is technical and is not used anywhere
    features.numeric_values = {"amount": "identity"}

    # name of target col
    features.target_col = "target_target_flag"

    ### TIME ###
    config.max_time = 17623.972627314815
    config.min_time = 17081.0

    # train specific parameters
    train = config.train = ml_collections.ConfigDict()
    # validation specific
    val = config.val = ml_collections.ConfigDict()

    # splitters
    train.split_strategy = {
        "split_strategy": "SampleSlices",
        "split_count": 5,
        "cnt_min": 15,
        "cnt_max": 150,
    }
    val.split_strategy = {
        "split_strategy": "SampleSlices",
        "split_count": 5,
        "cnt_min": 15,
        "cnt_max": 150,
    }

    # dropout
    train.dropout = 0.05
    train.max_seq_len = 100

    val.max_seq_len = 100

    train.num_workers = 1
    val.num_workers = 1

    train.batch_size = 4
    val.batch_size = 4

    return config
