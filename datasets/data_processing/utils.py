import yaml
import pyarrow.parquet as pq
import numpy as np


def read_yaml(path):
    with open(path, "r") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
    return yaml_data

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
