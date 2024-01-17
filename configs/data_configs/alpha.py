from pathlib import Path

import ml_collections


def data_configs():
    config = ml_collections.ConfigDict()

    ########## DATA ##############

    config.train_path = (
        Path(__file__).parent.parent.parent
        / "experiments"
        / "alpha"
        / "data"
        / "train_new.parquet"
    )
    config.test_path = (
        Path(__file__).parent.parent.parent
        / "experiments"
        / "alpha"
        / "data"
        / "test_new.parquet"
    )

    config.load_distributed = True
    config.recon_limit = 10000
    config.gen_limit = 10000
    config.predict_limit = 100000
    config.FT_number_objects = [1000, 100000]
    config.post_gen_FT_epochs = 5
    config.pre_trained_contrastive_path = "alpha/logs/CONTRASTIVE_GRU512-32emb/seed_0/ckpt/CONTRASTIVE_GRU512-32emb/seed_0/epoch__0040.ckpt"
    config.track_metric = "roc_auc"

    config.client_list_shuffle_seed = (
        0  # seed for splitting data to train and validation
    )
    config.valid_size = 0.05  # validation size
    config.test_size = 0.0  # pinch_test size
    config.col_id = "seq_id"  # column defining ids. used for sorting data

    features = config.features = ml_collections.ConfigDict()
    # dict below should define all the features that are not numeric with names as keys.
    # "in" parameter is used to clip values at the input.
    config.shift_embedding = False  # embeddings start with 1
    features.embeddings = {
        "currency": {"in": 12, "out": 12, "max_value": 12},
        "operation_kind": {"in": 8, "out": 8, "max_value": 8},
        "card_type": {"in": 174, "out": 174, "max_value": 174},
        "operation_type": {"in": 23, "out": 23, "max_value": 23},
        "operation_type_group": {"in": 5, "out": 5, "max_value": 5},
        "ecommerce_flag": {"in": 4, "out": 4, "max_value": 4},
        "payment_system": {"in": 8, "out": 8, "max_value": 8},
        "income_flag": {"in": 4, "out": 4, "max_value": 4},
        "mcc": {"in": 109, "out": 109, "max_value": 109},
        "country": {"in": 25, "out": 25, "max_value": 25},
        "city": {"in": 164, "out": 164, "max_value": 164},
        "mcc_category": {"in": 29, "out": 29, "max_value": 29},
        "day_of_week": {"in": 8, "out": 8, "max_value": 8},
        "hour": {"in": 25, "out": 25, "max_value": 25},
        "weekofyear": {"in": 54, "out": 54, "max_value": 54},
        "product": {"in": 6, "out": 6, "max_value": 6},
    }
    # all numeric features are defined here as keys
    features.numeric_values = {
        "amnt": "identity",
        "hour_diff": "identity",
        "days_before": "identity",
    }

    # name of target col
    features.target_col = "flag"
    config.num_classes = 2

    ### TIME ###
    config.max_time = 1.0
    config.min_time = 0.0

    # train specific parameters
    train = config.train = ml_collections.ConfigDict()
    # validation specific
    val = config.val = ml_collections.ConfigDict()
    test = config.test = ml_collections.ConfigDict()

    # splitters
    train.split_strategy = {
        "split_strategy": "SampleSlices",
        "split_count": 5,
        "cnt_min": 50,
        "cnt_max": 150,
    }
    val.split_strategy = {
        "split_strategy": "SampleSlices",
        "split_count": 5,
        "cnt_min": 15,
        "cnt_max": 150,
    }

    # train.split_strategy = {"split_strategy": "NoSplit"}
    # val.split_strategy = {"split_strategy": "NoSplit"}
    test.split_strategy = {"split_strategy": "NoSplit"}

    # dropout
    train.dropout = 0.0
    config.use_constant_pad = False
    train.max_seq_len = 200
    test.max_seq_len = 200
    val.max_seq_len = 200

    train.num_workers = 1
    val.num_workers = 1
    test.num_workers = 1

    train.batch_size = 256
    val.batch_size = 256
    test.batch_size = 16

    return config
