import ml_collections


def data_configs():
    config = ml_collections.ConfigDict()

    ########## DATA ##############
    
    config.preprocessor_path = None
    config.features_emb_dim = 16


    return config
