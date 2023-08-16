import ml_collections


def data_configs():
    config = ml_collections.ConfigDict()

    # data
    config.train_path = '~/event_seq/experiments/rosbank/data/train_trx.parquet'
    config.client_list_shuffle_seed = 0xAB0BA #seed for splitting data to train and validation
    config.valid_size = 0.05 #validation size 
    config.col_id = 'cl_id' #column defining ids. used for sorting data

    features = config.features = ml_collections.ConfigDict()
    # dict below should define all the features that are not numeric with names as keys. 
    # "in" parameter is used to clip values at the input.
    # have not figured out the purpose of "out"
    features.embeddings = {
        "mcc": {"in": 100, "out": 24},
        "channel_type": {"in": 4, "out": 4},
        "currency": {"in": 4, "out": 4},
        "trx_category": {"in": 10, "out": 4}
      }
    # all numeric features are defined here as keys
    # seem like its value is technical and is not used anywhere
    features.numeric_values = {
        "amount": "identity"
      }

    #splitters
    config.split_strategy = {
        "split_strategy": "SampleSlices",
        "split_count": 5,
        "cnt_min": 15,
        "cnt_max": 150
      }

    return config