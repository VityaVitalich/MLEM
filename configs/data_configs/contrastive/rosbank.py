from pathlib import Path

import ml_collections


def data_configs():
    config = ml_collections.ConfigDict()

    ########## DATA ##############

    config.train_path = (
        Path(__file__).parent.parent.parent.parent
        / "experiments"
        / "rosbank"
        / "data"
        / "train_trx.parquet"
    )
    config.test_path = (
        Path(__file__).parent.parent.parent.parent
        / "experiments"
        / "rosbank"
        / "data"
        / "test_trx.parquet"
    )
    config.load_distributed = False
    config.FT_number_objects = [1000, 'all']
    config.post_gen_FT_epochs = 25
    
    config.track_metric = "roc_auc"

    config.client_list_shuffle_seed = (
        0  # seed for splitting data to train and validation
    )
    config.valid_size = 0.05  # validation size
    config.test_size = 0.0  # pinch_test size
    config.col_id = "cl_id"  # column defining ids. used for sorting data

    features = config.features = ml_collections.ConfigDict()
    # dict below should define all the features that are not numeric with names as keys.
    # "in" parameter is used to clip values at the input.
    features.embeddings = {
        "mcc": {"in": 100, "out": 24, "max_value": 100},
        "channel_type": {"in": 4, "out": 4, "max_value": 5},
        "currency": {"in": 4, "out": 4, "max_value": 5},
        "trx_category": {"in": 10, "out": 4, "max_value": 12},
    }
    # all numeric features are defined here as keys
    features.numeric_values = {"amount": "identity"}

    # name of target col
    features.target_col = "target_target_flag"
    config.num_classes = 2

    ### TIME ###
    config.max_time = 17623.972627314815
    config.min_time = 17081.0

    # train specific parameters
    train = config.train = ml_collections.ConfigDict()
    # validation specific
    val = config.val = ml_collections.ConfigDict()
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
    config.use_constant_pad = False
    train.max_seq_len = 200
    test.max_seq_len = 200
    val.max_seq_len = 200

    train.num_workers = 1
    val.num_workers = 1
    test.num_workers = 1

    train.batch_size = 128
    val.batch_size = 128
    test.batch_size = 16

    return config
