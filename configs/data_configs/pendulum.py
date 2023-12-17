from pathlib import Path

import ml_collections


def data_configs():
    config = ml_collections.ConfigDict()

    ########## DATA ##############

    config.train_path = (
        Path(__file__).parent.parent.parent
        / "experiments"
        / "pendulum"
        / "data"
        / "train_hawkes_16.parquet"  # "train_trx.parquet"
    )
    config.test_path = (
        Path(__file__).parent.parent.parent
        / "experiments"
        / "pendulum"
        / "data"
        / "test_hawkes_16.parquet"
    )
    config.load_distributed = False
    config.recon_limit = 100
    config.gen_limit = 100
    config.num_plots = 15

    config.FT_number_objects = []
    config.post_gen_FT_epochs = 10

    config.track_metric = "mse"

    config.client_list_shuffle_seed = (
        0x3AB0D  # 0xAB0BA  # seed for splitting data to train and validation
    )
    config.valid_size = 0.1  # validation size
    config.test_size = 0.0  # pinch_test size
    config.col_id = "pendulum_id"  # column defining ids. used for sorting data

    features = config.features = ml_collections.ConfigDict()
    # dict below should define all the features that are not numeric with names as keys.
    # "in" parameter is used to clip values at the input.
    # have not figured out the purpose of "out"
    features.embeddings = {
        str(i):  {"in": 2, "out": 1, "max_value": 2} for i in range(0, 256)
        #"pad_category": {"in": 2, "out": 1, "max_value": 2},
    }
    # all numeric features are defined here as keys
    # seem like its value is technical and is not used anywhere
    features.numeric_values = {}#{str(i): "identity" for i in range(0, 256)}

    # name of target col
    features.target_col = "flag"
    config.num_classes = 1

    ### TIME ###
    config.max_time = 5.0
    config.min_time = 0.0

    # train specific parameters
    train = config.train = ml_collections.ConfigDict()
    # validation specific
    val = config.val = ml_collections.ConfigDict()
    test = config.test = ml_collections.ConfigDict()

    # splitters
    # train.split_strategy = {
    #     "split_strategy": "SampleSlices",
    #     "split_count": 5,
    #     "cnt_min": 50,
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
    test.split_strategy = {"split_strategy": "NoSplit"}

    # dropout
    train.dropout = 0.0
    config.use_constant_pad = False
    train.max_seq_len = 100
    test.max_seq_len = 100
    val.max_seq_len = 100

    train.num_workers = 1
    val.num_workers = 1
    test.num_workers = 1

    train.batch_size = 128
    val.batch_size = 128
    test.batch_size = 16

    return config
