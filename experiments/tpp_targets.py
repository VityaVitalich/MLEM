import pyspark
from argparse import ArgumentParser
from pathlib import Path
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np

def reload_with_new_targets(train_path, test_path, event_column, Ns=[1]):
    for path in [train_path, test_path]:
        path = Path(path)
        new_path = path.with_name("tpp_" + path.name)
        # spark = SparkSession.builder.appName("reload_targets").getOrCreate()
        df = pd.read_parquet(path)
        cut_n = max(Ns)
        cols_to_cut = [col for col in df if isinstance(df.iloc[0][col], np.ndarray)]
        for n in Ns:
            df[f"{n}_time"] = df["event_time"].apply(lambda x: x[-cut_n + (n - 1)])
            df[f"{n}_event"] = df[event_column].apply(lambda x: x[-cut_n + (n - 1)])

        for col in cols_to_cut:
            df[col] = df[col].apply(lambda x: x[:-cut_n])
            assert df[col].apply(len).min() > cut_n
        return 
        print("Original DataFrame:")
        df.show()
        if path == train_path and "alpha" in path:
            num_partitions = 100
            df = df.repartition(num_partitions)
        print("DataFrame after:")
        df.show()
        # df.write.parquet(new_path, mode="overwrite")
        spark.stop()

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
        train_path = "./age/data/train.parquet"
        test_path = "./age/data/test.parquet"
        event_column = "mcc"
        Ns = [1, 10, 30]
    else:
        raise NotImplementedError

    reload_with_new_targets(train_path, test_path, event_column, Ns)