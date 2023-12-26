import pyspark
from argparse import ArgumentParser
from pathlib import Path
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
import pyarrow as pa


def reload_with_new_targets(train_path, test_path, event_column, Ns=[1]):
    for path in [train_path, test_path]:
        path = Path(path)
        new_path = path.with_name("tpp_" + path.name)
        df = pd.read_parquet(path)
        print("Data loaded")
        cut_n = max(Ns)
        cols_to_cut = [col for col in df if isinstance(df.iloc[0][col], np.ndarray)]
        for n in Ns:
            df[f"{n}_time"] = df["event_time"].apply(lambda x: x[-cut_n + (n - 1)])
            df[f"{n}_event"] = df[event_column].apply(lambda x: x[-cut_n + (n - 1)])
        print("Targets added")
        for col in cols_to_cut:
            df[col] = df[col].apply(lambda x: x[:-cut_n])
            assert df[col].apply(len).min() > 0
        print("Trunsactions cuted")
        if str(path) == train_path and "alpha" in str(path):
            n_partition = 100
        else:
            n_partition = 10
        df["partition_idx"] = np.random.choice(range(n_partition), size=df.shape[0])
        table = pa.Table.from_pandas(df, preserve_index=False)
        pa.parquet.write_to_dataset(table, root_path=new_path, partition_cols=["partition_idx"])

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dataset", help="dataset name", default="age")
    args = parser.parse_args()
    if args.dataset == "age":
        train_path = "./age/data/train_trx.parquet"
        test_path = "./age/data/test_trx.parquet"
        event_column = "small_group"
        Ns = [1, 10, 30, 100]
    elif args.dataset == "alpha":
        train_path = "./alpha/data/train.parquet"
        test_path = "./alpha/data/test.parquet"
        event_column = "mcc"
        Ns = [1, 5]
    else:
        raise NotImplementedError

    reload_with_new_targets(train_path, test_path, event_column, Ns)