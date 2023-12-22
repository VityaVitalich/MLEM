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
        print(df.iloc[0]["event_time"], cols_to_cut)
        for col in cols_to_cut:
            df[col] = df[col].apply(lambda x: x[:-cut_n])
        print(df.iloc[0]["event_time"])
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
        Ns = [1, 10, 100]
    elif args.dataset == "alpha":
        train_path = "./age/data/train.parquet"
        test_path = "./age/data/test.parquet"
        event_column = "mcc"
        Ns = [1, 10, 100]
    else:
        raise NotImplementedError

    reload_with_new_targets(train_path, test_path, event_column, Ns)