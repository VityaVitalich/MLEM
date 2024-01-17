from pathlib import Path

import ml_collections


def data_configs():
    config = ml_collections.ConfigDict()

    ########## DATA ##############

    config.train_path = (
        Path(__file__).parent.parent.parent
        / "experiments"
        / "age"
        / "data"
        / "train_trx.parquet"
    )

    config.test_path = (
        Path(__file__).parent.parent.parent
        / "experiments"
        / "age"
        / "data"
        / "test_trx.parquet"
    )

    config.load_distributed = False
    config.FT_number_objects = [1000, "all"]
    config.post_gen_FT_epochs = 10
    config.pre_trained_contrastive_path = "age/logs/CONTRASTIVE_GRU512-32emb/seed_2/ckpt/CONTRASTIVE_GRU512-32emb/seed_2/epoch__0100.ckpt"
    config.track_metric = "accuracy"

    config.client_list_shuffle_seed = (
        1   # seed for splitting data to train and validation
    )
    config.valid_size = 0.05  # validation size
    config.test_size = 0.0  # pinch_test size
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

    # # splitters
    # train.split_strategy = {
    #     "split_strategy": "SampleSlices",  # SampleSlices
    #     "split_count": 5,
    #     # "seq_len": 25,
    #     "cnt_min": 25,
    #     "cnt_max": 200,
    # }
    # val.split_strategy = {
    #     "split_strategy": "SampleSlices",  # SampleSlices
    #     "split_count": 5,
    #     #  "seq_len": 50,
    #     "cnt_min": 25,
    #     "cnt_max": 100,
    # }
    train.split_strategy = {"split_strategy": "NoSplit"}
    val.split_strategy = {"split_strategy": "NoSplit"}
    test.split_strategy = {"split_strategy": "NoSplit"}

    # dropout
    train.dropout = 0.01

    # seq len
    config.min_seq_len = 25
    config.use_constant_pad = False
    train.max_seq_len = 1000
    val.max_seq_len = 1000
    test.max_seq_len = 1000

    train.num_workers = 1
    val.num_workers = 1
    test.num_workers = 1

    train.batch_size = 128
    val.batch_size = 128
    test.batch_size = 16

    return config
