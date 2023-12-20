import random
from tqdm import tqdm
import numpy as np
import pyarrow.parquet as pq


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


def prepare_embeddings(seq, conf):
    embeddings = list(conf.features.embeddings.keys())

    feature_keys = embeddings + list(conf.features.numeric_values.keys())

    for rec in tqdm(seq):
        if "feature_arrays" in rec:
            feature_arrays = rec["feature_arrays"]
            feature_arrays = {
                k: v for k, v in feature_arrays.items() if k in feature_keys
            }
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
        # feature_arrays = {
        #     k: v + (1 if k in embeddings else 0) for k, v in feature_arrays.items()
        # }

        # clip embeddings dictionary by max value
        for e_name, e_params in conf.features.embeddings.items():
            feature_arrays[e_name] = feature_arrays[e_name].clip(0, e_params["in"] - 1)

        normed_time = (np.array(rec["event_time"]) - conf.min_time) / (
            conf.max_time - conf.min_time
        )
        feature_arrays["event_time"] = normed_time
        rec["event_time"] = normed_time

        rec["feature_arrays"] = feature_arrays
        yield rec


def shuffle_client_list_reproducible(conf, data):
    dataset_col_id = conf.get("col_id", "client_id")
    data = sorted(data, key=lambda x: x.get(dataset_col_id))  # changed from COLES a bit
    random.Random(conf.client_list_shuffle_seed).shuffle(data)
    return data


def prepare_data(conf, supervised, pinch_test=False):
    train_path = conf.train_path

    data = read_pyarrow_file(train_path)
    data = data

    data = prepare_embeddings(data, conf)
    data = shuffle_client_list_reproducible(conf, data)
    data = list(data)
    train_data, valid_data, test_data = split_dataset(
        data, conf, supervised, pinch_test
    )
    print(
        "Data shapes: train %d, val %d, test %d"
        % (len(train_data), len(valid_data), len(test_data))
    )
    return train_data, valid_data, test_data


def split_dataset(data, conf, supervised, pinch_test=False):
    min_seq_len = conf.get("min_seq_len", 5)

    labeled_ix = []
    unlabeled_ix = []
    for i, rec in enumerate(data):
        if rec[conf.features.target_col] is not None and not np.isnan(
            float(rec[conf.features.target_col])
        ):
            labeled_ix.append(i)
        else:
            unlabeled_ix.append(i)
    random.Random(conf.client_list_shuffle_seed).shuffle(labeled_ix)
    random.Random(conf.client_list_shuffle_seed).shuffle(unlabeled_ix)

    test_size = 0 if not pinch_test else int(len(labeled_ix) * conf.test_size)
    val_labeled_size = int(len(labeled_ix) * conf.valid_size)
    val_unlabeled_size = int(len(unlabeled_ix) * conf.valid_size)

    test_ix = labeled_ix[:test_size]
    val_labeled_ix = labeled_ix[test_size : test_size + val_labeled_size]
    train_labeled_ix = labeled_ix[test_size + val_labeled_size :]

    val_unlabeled_ix = unlabeled_ix[:val_unlabeled_size]
    train_unlabeled_ix = unlabeled_ix[val_unlabeled_size:]

    assert len(train_unlabeled_ix) + len(train_labeled_ix) + len(
        val_unlabeled_ix
    ) + len(val_labeled_ix) + len(test_ix) == len(data)
    if supervised:
        train_unlabeled_ix = []
        val_unlabeled_ix = []
    train_data = [
        data[i]
        for i in train_unlabeled_ix + train_labeled_ix
        if len(data[i]["event_time"]) >= min_seq_len
    ]

    valid_data = [
        data[i]
        for i in val_unlabeled_ix + val_labeled_ix
        if len(data[i]["event_time"]) >= min_seq_len
    ]
    test_data = [data[i] for i in test_ix]
    return train_data, valid_data, test_data


def prepare_test_data(conf):
    data = read_pyarrow_file(conf.test_path)
    data = data

    data = prepare_embeddings(data, conf)
    #   data = shuffle_client_list_reproducible(conf, data)
    data = list(data)

    return data
