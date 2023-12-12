import random

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm


def read_pyarrow_file(path, use_threads=True):
    p_table = pq.read_table(
        source=path,
        use_threads=use_threads,
    )

    col_indexes = [n for n in p_table.column_names]

    def get_records():
        for rb in p_table.to_batches():
            col_arrays = [rb.column(i) for i, _ in enumerate(col_indexes)]
            col_arrays = [a.to_numpy(zero_copy_only=False) for a in col_arrays]
            for row in zip(*col_arrays):
                # np.array(a) makes `a` writable for future usage
                rec = {
                    n: np.array(a) if isinstance(a, np.ndarray) else a
                    for n, a in zip(col_indexes, row)
                }
                yield rec

    return get_records()


def process_record(rec, conf, feature_keys, embeddings):
    if "feature_arrays" in rec:
        feature_arrays = rec["feature_arrays"]
        feature_arrays = {k: v for k, v in feature_arrays.items() if k in feature_keys}
    else:
        feature_arrays = {k: v for k, v in rec.items() if k in feature_keys}

    # TODO: datetime processing. Take date-time features

    # shift embeddings to 1, 0 is padding value
    shift_values = conf.get("shift_embedding", True)
    new_feature_arrays = {}
    for k, v in feature_arrays.items():
        value = v
        if (k in embeddings) and shift_values:
            value += 1

        new_feature_arrays[k] = value

    feature_arrays = new_feature_arrays

    # clip embeddings dictionary by max value
    for e_name, e_params in conf.features.embeddings.items():
        feature_arrays[e_name] = feature_arrays[e_name].clip(0, e_params["in"] - 1)

    normed_time = (np.array(rec["event_time"]) - conf.min_time) / (
        conf.max_time - conf.min_time
    )
    # print(normed_time)
    feature_arrays["event_time"] = normed_time
    rec["event_time"] = normed_time

    rec["feature_arrays"] = feature_arrays
    return rec


def prepare_embeddings(seq, conf, is_train):
    min_seq_len = 5
    embeddings = list(conf.features.embeddings.keys())

    feature_keys = embeddings + list(conf.features.numeric_values.keys())

    for rec in seq:
        seq_len = len(rec["event_time"])
        if is_train and seq_len < min_seq_len:
            continue
        yield process_record(rec, conf, feature_keys, embeddings)


def shuffle_client_list_reproducible(conf, data):
    if conf.client_list_shuffle_seed != 0:
        dataset_col_id = conf.get("col_id", "client_id")
        data = sorted(
            data, key=lambda x: x.get(dataset_col_id)
        )  # changed from COLES a bit
        random.Random(conf.client_list_shuffle_seed).shuffle(data)
    return data


def prepare_data_dist(conf, supervised):
    train_path = conf.train_path

    data = read_pyarrow_file(train_path)
    if supervised:
        data = (rec for rec in data if rec[conf.features.target_col] is not None)
        data = (
            rec for rec in data if not np.isnan(float(rec[conf.features.target_col]))
        )
    data = tqdm(data)

    data = prepare_embeddings(data, conf, is_train=True)
    data = shuffle_client_list_reproducible(conf, data)
    data = list(data)

    valid_ix = np.arange(len(data))
    valid_ix = np.random.choice(
        valid_ix, size=int(len(data) * conf.valid_size), replace=False
    )
    valid_ix = set(valid_ix.tolist())

    # logger.info(f'Loaded {len(data)} rows. Split in progress...')
    train_data = [rec for i, rec in enumerate(data) if i not in valid_ix]
    valid_data = [rec for i, rec in enumerate(data) if i in valid_ix]

    return train_data, valid_data


def prepare_test_data_dist(conf):
    data = read_pyarrow_file(conf.test_path)
    data = tqdm(data)

    data = prepare_embeddings(data, conf, is_train=False)
    #   data = shuffle_client_list_reproducible(conf, data)
    data = list(data)

    return data
