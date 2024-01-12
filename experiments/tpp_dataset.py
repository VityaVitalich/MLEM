from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import numpy as np
import os
import pyarrow as pa

def row_drop(row, drop_rate):   
    curr_len = row.iloc[0].shape[0]
    keep_len = int(curr_len * drop_rate)
    indices_to_drop = np.random.choice(curr_len, keep_len, replace=False)
    row = row.apply(lambda x: np.delete(x, indices_to_drop))
    return row

def row_permute(row):   
    curr_len = row.iloc[0].shape[0]
    new_ids = np.random.permutation(curr_len)
    row = row.apply(lambda x: x[new_ids])
    return row

def reload_with_new_targets(train_path, test_path, event_column, Ns=[1]):
    for path in [train_path, test_path]:
        path = Path(path)
        new_path = path.with_name("tpp_" + path.name)
        df = pd.read_parquet(path)
        print("Data loaded")
        cut_n = max(Ns)
        df = df[df['trx_count'] > cut_n]
        cols_to_cut = [col for col in df if isinstance(df.iloc[0][col], np.ndarray)]
        for n in Ns:
            df[f"{n}_time"] = df["event_time"].apply(lambda x: x[-cut_n + (n - 1)])
            if event_column:
                df[f"{n}_event"] = df[event_column].apply(lambda x: x[-cut_n + (n - 1)])
        print("Targets added")
        for col in cols_to_cut:
            df[col] = df[col].apply(lambda x: x[:-cut_n])
            assert df[col].apply(len).min() > 0
        print("Transactions cuted")
        if str(path) == train_path:
            n_partition = 100
        else:
            n_partition = 10
        df["partition_idx"] = np.random.choice(range(n_partition), size=df.shape[0])
        table = pa.Table.from_pandas(df, preserve_index=False)
        pa.parquet.write_to_dataset(table, root_path=new_path, partition_cols=["partition_idx"])

def reload_drop(train_path, test_path):
    for drop in [0.1, 0.3, 0.5, 0.7]:
        for path in [train_path, test_path]:
            print(f"Drop {drop} started for {path}")
            path = Path(path)
            new_path = path.with_name(f"Drop_{drop}_" + path.name)
            df = pd.read_parquet(path)
            print("Data loaded")
            cols_to_sample = [col for col in df if isinstance(df.iloc[0][col], np.ndarray)]
            print(cols_to_sample)
            df[cols_to_sample] = df[cols_to_sample].apply(lambda row: row_drop(row, drop), axis=1)
            
            print("Dropout done")
            if str(path) == train_path and "alpha" in str(path):
                n_partition = 100
            else:
                n_partition = 10
            df["partition_idx"] = np.random.choice(range(n_partition), size=df.shape[0])
            table = pa.Table.from_pandas(df, preserve_index=False)
            pa.parquet.write_to_dataset(table, root_path=new_path, partition_cols=["partition_idx"])

def reload_permute(train_path, test_path):
    for path in [train_path, test_path]:
        print(f"Permute started for {path}")
        path = Path(path)
        new_path = path.with_name(f"Permute_" + path.name)
        df = pd.read_parquet(path)
        print("Data loaded")
        cols_to_permute = [col for col in df if isinstance(df.iloc[0][col], np.ndarray)]
        print(cols_to_permute)
        df[cols_to_permute] = df[cols_to_permute].apply(lambda row: row_permute(row), axis=1)
        
        print("Permute done")
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
        Ns = [1, 2]
    elif args.dataset == "alpha":
        train_path = "./alpha/data/train_new.parquet"
        test_path = "./alpha/data/test_new.parquet"
        event_column = "mcc"
        Ns = [1, 2]
    elif args.dataset == "rosbank":
        train_path = "./rosbank/data/train_trx.parquet"
        test_path = "./rosbank/data/test_trx.parquet"
        event_column = "mcc"
        Ns = [1, 2]
    elif args.dataset == "physionet":
        train_path = "./physionet/data/train_trx.parquet"
        test_path = "./physionet/data/test_trx.parquet"
        event_column = None
        Ns = [1, 2]
    elif args.dataset == "pendulum_coord":
        train_path = "./pendulum/data/train_hawkes_coordinate.parquet"
        test_path = "./pendulum/data/test_hawkes_coordinate.parquet"
        event_column = None
        Ns = [1, 2]
    elif args.dataset == "pendulum_coord_100k":
        train_path = "./pendulum/data/train_hawkes_coordinate_100k.parquet"
        test_path = "./pendulum/data/test_hawkes_coordinate_100k.parquet"
        event_column = None
        Ns = [1, 2]
    elif args.dataset == "pendulum_image":
        train_path = "./pendulum/data/train_hawkes_16.parquet"
        test_path = "./pendulum/data/test_hawkes_16.parquet"
        event_column = None
        Ns = [1, 2]
    elif args.dataset == "amex":
        train_path = "./amex/data/train.parquet"
        test_path = "./amex/data/test.parquet"
        event_column = "D_120"
        Ns = [1, 2]
    elif args.dataset == "taobao":
        train_path = "./taobao/data/train.parquet"
        test_path = "./taobao/data/test.parquet"
        event_column = "item_category"
        Ns = [1, 2]
    else:
        raise NotImplementedError

    reload_with_new_targets(train_path, test_path, event_column, Ns)
    reload_drop(train_path, test_path)
    reload_permute(train_path, test_path)