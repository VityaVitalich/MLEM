import ml_collections


def data_configs():
    config = ml_collections.ConfigDict()

    # data
    config.train_path = '~/event_seq/experiments/rosbank/data/train_trx.parquet'

    return config