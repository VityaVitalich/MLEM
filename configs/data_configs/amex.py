from pathlib import Path

import ml_collections


def data_configs():
    config = ml_collections.ConfigDict()

    ########## DATA ##############

    config.train_path = (
        Path(__file__).parent.parent.parent
        / "experiments"
        / "amex"
        / "data"
        / "train.parquet"
    )
    config.test_path = (
        Path(__file__).parent.parent.parent
        / "experiments"
        / "amex"
        / "data"
        / "test.parquet"
    )

    config.load_distributed = True
    config.recon_limit = 10000
    config.gen_limit = 10000
    config.predict_limit = 100000
    config.FT_number_objects = [1000, 100000]
    config.post_gen_FT_epochs = 10
    # config.pre_trained_contrastive_path = "amex/logs/CONTRASTIVE_GRU512-32emb/seed_0/ckpt/CONTRASTIVE_GRU512-32emb/seed_0/epoch__0040.ckpt"
    config.track_metric = "accuracy"

    config.client_list_shuffle_seed = (
        0 # seed for splitting data to train and validation
    )
    config.valid_size = 0.05  # validation size
    config.test_size = 0.0  # pinch_test size
    config.col_id = "seq_id"  # column defining ids. used for sorting data

    features = config.features = ml_collections.ConfigDict()
    # dict below should define all the features that are not numeric with names as keys.
    # "in" parameter is used to clip values at the input.
    config.shift_embedding = False  # embeddings start with 1
    features.embeddings = {
        "B_30": {"in": 5, "out": 5, "max_value": 5},
        "B_38": {"in": 9, "out": 9, "max_value": 9},
        "D_114": {"in": 4, "out": 4, "max_value": 4},
        "D_116": {"in": 4, "out": 4, "max_value": 4},
        "D_117": {"in": 9, "out": 9, "max_value": 9},
        "D_120": {"in": 4, "out": 4, "max_value": 4},
        "D_126": {"in": 5, "out": 5, "max_value": 5},
        "D_63": {"in": 7, "out": 7, "max_value": 7},
        "D_64": {"in": 6, "out": 6, "max_value": 6},
        "D_66": {"in": 4, "out": 4, "max_value": 4},
        "D_68": {"in": 9, "out": 9, "max_value": 9},

    }
    # all numeric features are defined here as keys
    features.numeric_values = {
        "P_2": "identity",
        "D_39": "identity",
        "B_1": "identity",
        "B_2": "identity",
        "R_1": "identity",
        "S_3": "identity",
        "D_41": "identity",
        "B_3": "identity",
        "D_42": "identity",
        "D_43": "identity",
        "D_44": "identity",
        "B_4": "identity",
        "D_45": "identity",
        "B_5": "identity",
        "R_2": "identity",
        "D_46": "identity",
        "D_47": "identity",
        "D_48": "identity",
        "D_49": "identity",
        "B_6": "identity",
        "B_7": "identity",
        "B_8": "identity",
        "D_50": "identity",
        "D_51": "identity",
        "B_9": "identity",
        "R_3": "identity",
        "D_52": "identity",
        "P_3": "identity",
        "B_10": "identity",
        "D_53": "identity",
        "S_5": "identity",
        "B_11": "identity",
        "S_6": "identity",
        "D_54": "identity",
        "R_4": "identity",
        "S_7": "identity",
        "B_12": "identity",
        "S_8": "identity",
        "D_55": "identity",
        "D_56": "identity",
        "B_13": "identity",
        "R_5": "identity",
        "D_58": "identity",
        "S_9": "identity",
        "B_14": "identity",
        "D_59": "identity",
        "D_60": "identity",
        "D_61": "identity",
        "B_15": "identity",
        "S_11": "identity",
        "D_62": "identity",
        "D_65": "identity",
        "D_135": "identity",
        "D_136": "identity",
        "B_16": "identity",
        "B_17": "identity",
        "B_18": "identity",
        "B_19": "identity",
        "B_20": "identity",
        "S_12": "identity",
        "R_6": "identity",
        "S_13": "identity",
        "B_21": "identity",
        "D_69": "identity",
        "B_22": "identity",
        "R_26": "identity",
        "R_27": "identity",
        "D_70": "identity",
        "D_71": "identity",
        "D_72": "identity",
        "S_15": "identity",
        "B_23": "identity",
        "D_73": "identity",
        "P_4": "identity",
        "D_74": "identity",
        "D_75": "identity",
        "D_76": "identity",
        "B_24": "identity",
        "R_7": "identity",
        "D_77": "identity",
        "B_25": "identity",
        "B_26": "identity",
        "D_78": "identity",
        "D_79": "identity",
        "R_8": "identity",
        "R_9": "identity",
        "S_16": "identity",
        "D_80": "identity",
        "R_10": "identity",
        "R_11": "identity",
        "B_27": "identity",
        "D_81": "identity",
        "D_82": "identity",
        "S_17": "identity",
        "R_12": "identity",
        "B_28": "identity",
        "R_13": "identity",
        "D_83": "identity",
        "R_14": "identity",
        "R_15": "identity",
        "D_84": "identity",
        "R_16": "identity",
        "B_29": "identity",
        "S_18": "identity",
        "D_86": "identity",
        "D_134": "identity",
        "D_87": "identity",
        "R_17": "identity",
        "R_18": "identity",
        "D_88": "identity",
        "B_31": "identity",
        "S_19": "identity",
        "R_19": "identity",
        "B_32": "identity",
        "S_20": "identity",
        "R_20": "identity",
        "R_21": "identity",
        "B_33": "identity",
        "D_89": "identity",
        "R_22": "identity",
        "R_23": "identity",
        "D_91": "identity",
        "D_92": "identity",
        "D_93": "identity",
        "D_94": "identity",
        "R_24": "identity",
        "R_25": "identity",
        "D_96": "identity",
        "S_22": "identity",
        "S_23": "identity",
        "S_24": "identity",
        "S_25": "identity",
        "S_26": "identity",
        "D_102": "identity",
        "D_103": "identity",
        "D_104": "identity",
        "D_105": "identity",
        "D_106": "identity",
        "D_107": "identity",
        "B_36": "identity",
        "B_37": "identity",
        "D_143": "identity",
        "D_144": "identity",
        "D_145": "identity",
        "D_108": "identity",
        "D_109": "identity",
        "D_110": "identity",
        "D_111": "identity",
        "B_39": "identity",
        "D_112": "identity",
        "B_40": "identity",
        "S_27": "identity",
        "D_113": "identity",
        "D_115": "identity",
        "D_141": "identity",
        "D_142": "identity",
        "D_118": "identity",
        "D_119": "identity",
        "D_121": "identity",
        "D_122": "identity",
        "D_123": "identity",
        "D_124": "identity",
        "D_125": "identity",
        "D_127": "identity",
        "D_138": "identity",
        "D_139": "identity",
        "D_140": "identity",
        "D_128": "identity",
        "D_129": "identity",
        "B_41": "identity",
        "B_42": "identity",
        "D_130": "identity",
        "D_131": "identity",
        "D_132": "identity",
        "D_133": "identity",
        "D_137": "identity",
        "R_28": "identity",
    }

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
    train.max_seq_len = 13
    test.max_seq_len = 13
    val.max_seq_len = 13

    train.num_workers = 1
    val.num_workers = 1
    test.num_workers = 1

    train.batch_size = 256
    val.batch_size = 256
    test.batch_size = 16

    return config
