from torch.utils.data import Dataset
from utils import read_yaml, read_pyarrow_file
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import Window, types
from pyspark.sql import functions as F
from random import Random
import numpy as np
import pandas as pd

import os


class SequenceDataset(Dataset):
    def __init__(self, cfg):
        # 1. Create csv or parquet. Put it into desired folder.
        # Assert correct time encoding(unix timestamp prefered).
        # 2. Fill out yaml config.
        # 3. Create SequenceDataset, pass config. Call load_dataset().
        # Dataset will be brought to the right format,
        # splitted and saved in "data_path" folder.
        # 4. Call e.g. get_dataset(labeled, split="train")
        # conf['dataset']['col_id'] = self.cfg['index_column']
        # conf['dataset.train_path'] = self.cfg["data_path"]).parent / "train.parquet"
        # conf['dataset.test_path'] = self.cfg["data_path"]).parent / "test.parquet"
        # Pohui ne budet vala y vas zaebali.
        self.cfg = read_yaml(cfg) if isinstance(cfg, str) else cfg
        self.spark = None
        data_path = self.cfg["data_path"]
        self.train_path = Path(data_path).parent / "train.parquet"
        self.test_path = Path(data_path).parent / "test.parquet"
        assert isinstance(self.cfg["target_columns"], list), "target_columns has to be a list"

    def get_dataset(self, split="train", labeled_only=False):
        assert split in ["train", "test"]
        if split == "train":
            data = read_pyarrow_file(
                Path(self.cfg["data_path"]).parent / "train.parquet"
            )
            if labeled_only:
                data = (
                    rec for rec in data if rec[self.cfg["target_columns"][0]] is not None
                )
                data = (
                    rec for rec in data if not np.isnan(rec[self.cfg["target_columns"][0]])
                )
        elif split == "test":
            data = read_pyarrow_file(
                Path(self.cfg["data_path"]).parent / "test.parquet"
            )
        return data

    def load_dataset(self, shuffle=True):
        # Either parquet "data" folder or csv file
        data_path = self.cfg["data_path"]
        train_path = Path(data_path).parent / "train.parquet"
        test_path = Path(data_path).parent / "test.parquet"

        if train_path.exists():
            print(f"{str(train_path)} already exists.")
            return self
        if test_path.exists():
            print(f"{str(test_path)} already exists.")
            return self
        if self.spark is None or self.spark._jsc.sc().isStopped():
            self.spark = SparkSession.builder \
                .config("spark.driver.extraJavaOptions", "-Xss512m") \
                .config("spark.executor.extraJavaOptions", "-Xss512m") \
                .config("spark.driver.memory", "64g") \
                .config("spark.executor.memory", "64g") \
                .config("spark.driver.memoryOverhead", "16g") \
                .config("spark.executor.memoryOverhead", "16g") \
                .getOrCreate()
        train_data = self.read_preprocessed_data(data_path)
        test_data = None
        if self.cfg.get("test_path") and Path(self.cfg["test_path"]).exists():
            print("Using separate test.")
            test_data = self.read_preprocessed_data(self.cfg["test_path"])
        train_data, test_data = self.process_time([train_data, test_data])
        train_data.show()
        if not self.is_collected_list(train_data):
            print("Collecting lists...")
            train_data = self.collect_lists(train_data)
            test_data = self.collect_lists(test_data) if test_data else test_data
        if test_data is None:
            print("Splitting data into train and test...")
            train_data, test_data = self.split_dataset(train_data, shuffle)
        # train_data = train_data.repartition(100)
        # test_data = test_data.repartition(100)
        test_data.show(20)
        self.save_features(train_data, train_path)
        self.save_features(test_data, test_path)
        self.spark.stop()

        return self

    def split_dataset(self, all_data, shuffle=True):
        if self.spark is None or self.spark._jsc.sc().isStopped():
            self.spark = SparkSession.builder.getOrCreate()
        spark = self.spark
        s_clients = all_data.filter(F.col(self.cfg["target_columns"][0]).isNotNull())
        s_clients = set(
            cl[0]
            for cl in s_clients.select(self.cfg["index_column"]).distinct().collect()
        )

        # shuffle client list
        s_clients = sorted(s_clients)
        s_clients = [cl_id for cl_id in s_clients]
        if shuffle:
            Random(1).shuffle(s_clients)

        # split client list
        sizes = self.cfg["split_sizes"]
        n_train, n_test = sizes["train"], sizes["test"]
        assert n_train + n_test == 1.0, f"Wrong split sizes: {sizes}"

        n_train = int(len(s_clients) * n_train)
        n_test = int(len(s_clients) * n_test)
        # n_val = int(len(s_clients) * n_val)

        s_clients_train = s_clients[:n_train]
        s_clients_test = s_clients[n_train:]
        # s_clients_val = s_clients[n_test: n_test + n_val]

        s_clients_train = spark.createDataFrame(
            [(i,) for i in s_clients_train], [self.cfg["index_column"]]
        ).repartition(1)
        # s_clients_val = spark.createDataFrame([(i,) for i in s_clients_val],
        # [self.cfg["index_column"]])
        # .repartition(1)
        s_clients_test = spark.createDataFrame(
            [(i,) for i in s_clients_test], [self.cfg["index_column"]]
        ).repartition(1)
        s_clients = spark.createDataFrame(
            [(i,) for i in s_clients], [self.cfg["index_column"]]
        ).repartition(1)

        # split data
        labeled_train = all_data.join(
            s_clients_train, on=self.cfg["index_column"], how="inner"
        )
        labeled_test = all_data.join(
            s_clients_test, on=self.cfg["index_column"], how="inner"
        )
        # labeled_val = all_data.join(s_clients_val, on=self.cfg["index_column"]
        # , how='inner')

        unlabeled = all_data.join(
            s_clients, on=self.cfg["index_column"], how="left_anti"
        )
        train = labeled_train.union(unlabeled)
        test = labeled_test
        # val = labeled_val
        return train, test

    def collect_lists(self, df):
        # Input: pyspark df
        # Output: pyspark df, each row - one sequence
        col_id = self.cfg["index_column"]
        target_cols = self.cfg["target_columns"]
        col_list = [col for col in df.columns if col not in [col_id, *target_cols]]
        df = df.withColumn(
            "trx_count", F.count(F.lit(1)).over(Window.partitionBy(col_id))
        ).withColumn(
            "_rn",
            F.row_number().over(
                Window.partitionBy(col_id).orderBy(self.cfg["time_column"])
            ),
        )

        for col in col_list:
            df = df.withColumn(
                col, F.collect_list(col).over(Window.partitionBy(col_id).orderBy("_rn"))
            )
        df = df.filter("_rn = trx_count").drop("_rn")
        return df

    def process_time(self, dfs):
        print(f"Rename column {self.cfg['time_column']} with 'event_time'")
        self.print_stats(dfs)
        for i in range(len(dfs)):
            df = dfs[i]
            if df:
                df = df.withColumnRenamed(self.cfg["time_column"], "event_time")
                if isinstance(df.schema["event_time"].dataType, types.StringType):
                    raise TypeError("Bad time type(str)")
                    df = df.withColumn("event_time", F.unix_timestamp(
                        "event_time", "yyyy-MM-dd").cast(types.FloatType())
                    )
                dfs[i] = df
        self.cfg['time_column'] = "event_time"
        self.print_stats(dfs)
        return dfs
    
    def print_stats(self, dfs):
        for df in dfs:
            if df:
                print("Min/Max time:", df.agg(F.min(self.cfg['time_column']), F.max(self.cfg['time_column'])).first())
        return dfs

    def explode_list(self, df):
        # Input: pyspark df
        # Output: pyspark df, each row - one transaction
        pass

    def is_collected_list(self, df):
        value_counts = df.groupBy(self.cfg["index_column"]).agg(
            F.count("*").alias("count")
        )
        max_count = value_counts.orderBy(F.desc("count")).first()["count"]
        return max_count == 1

    def read_preprocessed_data(self, path):
        if self.spark is None or self.spark._jsc.sc().isStopped():
            self.spark = SparkSession.builder.getOrCreate()
        spark = self.spark
        ext = os.path.splitext(path)[1]
        if ext == ".parquet":
            return spark.read.parquet(path)
        elif ".csv" in path:
            return spark.read.csv(path, header=True, inferSchema=True)
        else:
            raise NotImplementedError(f'Unknown input file extension: "{ext}"')

    def save_features(self, df_data, save_path):
        df_data.write.parquet(str(save_path), mode="overwrite")
        print(f'Saved to: "{save_path}"')


class SberSequenceDataset(SequenceDataset):
    def load_dataset(self, train_ids, first_test_date):
        # Either parquet "data" folder or csv file
        data_path = self.cfg["data_path"]
        train_path = Path(data_path).parent / "train.parquet"
        test_path = Path(data_path).parent / "test.parquet"
        # val_path = Path(data_path).parent / "val.parquet"

        if train_path.exists():
            print(f"{str(train_path)} already exists.")
            return self
        if self.spark is None or self.spark._jsc.sc().isStopped():
            self.spark = SparkSession.builder.getOrCreate()

        data = self.read_preprocessed_data(data_path)
        if not self.is_collected_list(data):
            print("Collecting lists...")
            data = self.collect_lists(data)

        train, test = self.split_dataset(data, train_ids, first_test_date)
        self.save_features(train, train_path)
        self.save_features(test, test_path)
        # self.save_features(val, val_path)
        self.spark.stop()

        return self

    def split_dataset(self, all_data, train_ids, first_test_date):
        # def split_dataset(self, all_data, shuffle=True):
        if self.spark is None or self.spark._jsc.sc().isStopped():
            self.spark = SparkSession.builder.getOrCreate()
        spark = self.spark
        s_clients = all_data.filter(F.col(self.cfg["target_column"]).isNotNull())
        s_clients = (
            pd.Series(
                cl[0]
                for cl in s_clients.select(self.cfg["index_column"])
                .distinct()
                .collect()
            )
            .str.split("_", expand=True)
            .rename({0: "date", 1: "epk_id"}, axis=1)
            .assign(
                date=lambda df: pd.to_datetime(df["date"]),
                epk_id=lambda df: df["epk_id"].astype(int),
            )
        )
        train_clients = s_clients.query(
            "date < @first_test_date and epk_id.isin(@train_ids)"
        )
        test_clients = s_clients.query("date >= @first_test_date")
        s_clients_train = (
            train_clients["date"].astype(str)
            + "_"
            + train_clients["epk_id"].astype(str)
        ).tolist()
        s_clients_test = (
            test_clients["date"].astype(str) + "_" + test_clients["epk_id"].astype(str)
        ).tolist()

        s_clients_train = spark.createDataFrame(
            [(i,) for i in s_clients_train], [self.cfg["index_column"]]
        ).repartition(1)
        # s_clients_val = spark.createDataFrame([(i,) for i in s_clients_val],
        # [self.cfg["index_column"]])
        # .repartition(1)
        s_clients_test = spark.createDataFrame(
            [(i,) for i in s_clients_test], [self.cfg["index_column"]]
        ).repartition(1)
        # s_clients = spark.createDataFrame(
        #     [(i,) for i in s_clients], [self.cfg["index_column"]]
        # ).repartition(1)

        # split data
        labeled_train = all_data.join(
            s_clients_train, on=self.cfg["index_column"], how="inner"
        )
        labeled_test = all_data.join(
            s_clients_test, on=self.cfg["index_column"], how="inner"
        )
        # labeled_val = all_data.join(s_clients_val, on=self.cfg["index_column"]
        # , how='inner')

        # unlabeled = all_data.join(
        #     s_clients, on=self.cfg["index_column"], how="left_anti"
        # )
        train = labeled_train
        test = labeled_test
        # val = labeled_val
        return train, test


if __name__ == "__main__":
    cfg_path = "/home/dev/2023_03/Datasets/event_seq/datasets/configs/amex.yaml"
    sq = SequenceDataset(cfg_path)
    sq.load_dataset()
    print(next(sq.get_dataset(split="train")))
    print(next(sq.get_dataset(split="test")))
