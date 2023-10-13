import os
from pathlib import Path
from dataset import SequenceDataset
from utils import read_yaml
import numpy as np
from utils import read_pyarrow_file




import pandas as pd
import numpy as np
from tqdm import trange


random = np.random.default_rng(0xB0BA_C_3AB0DA)
start = pd.Timestamp("2021-01-01")
end = pd.Timestamp("2022-05-01")
dt = (end - start).total_seconds()


def gen_tx(n_tx, epk_id):
    tx_times = np.sort(random.uniform(high=dt, size=n_tx))
    evt_dttm = (tx_times.astype("timedelta64[s]") + start).astype("datetime64[ns]")
    report_date = (evt_dttm.astype("datetime64[M]") + np.timedelta64(1, "M") - np.timedelta64(1, "D")).astype("datetime64[ns]")
    df = pd.DataFrame(dict(
        epk_id=epk_id,
        report_date=report_date,
        evt_dttm=evt_dttm,
        trx_country=random.choice(["RUS", "FOREIGN"], size=n_tx, p=[.95, .05]),
        mcc_code=random.choice(np.arange(300, dtype=int), size=n_tx),
        trans_type=random.choice(np.arange(300, dtype=int), size=n_tx),
        card_cat_cd=random.choice(['CR', 'DB', 'MISSING'], size=n_tx, p=[.45, .45, .1]),
        ipt_name=random.choice(['MASTERCARD', 'MC', 'MIR', 'MISSING', 'SBER', 'VISA', 'ПРО100'], size=n_tx),
        iso_crncy_cd=random.choice(["RUS", "FOREIGN"], size=n_tx, p=[.95, .05]),
        ecom_fl=random.integers(low=0, high=2, size=n_tx),
        amt=random.exponential(1000, size=n_tx),
        trx_direction=random.choice(["C", "D"], size=n_tx, p=[.8, .2]),
    )).set_index(["epk_id", "report_date"])
    
    n_pl = len(np.unique(report_date))
    pl = (10 ** random.normal(loc=3, size=n_pl)) * random.choice([-1, 1], size=n_pl, p=[.3, .7])
    pl = pd.DataFrame(dict(
        epk_id=epk_id,
        report_date=np.unique(report_date),
        pl=pl,
    )).set_index(["epk_id", "report_date"])
    return df.join(pl)

def make_df(n_epk, save_path="data/synthetic/data.csv"):
    if Path(save_path).exists():
        print(f"File {save_path} already exists")
        return pd.read_csv(save_path)
    dfs = []
    for epk_id in trange(n_epk):
        n_tx = int(random.exponential(scale=1000))
        dfs.append(gen_tx(n_tx, epk_id))
    df = pd.concat(dfs, axis=0)
    dt = df.dtypes
    for c in df:
        if dt[c] in (np.dtype("O"), np.dtype("int64")):
            df[c] = df[c].astype("category")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path)
    return df

def preprocess(load_path, save_path="data/synthetic/processed/data.csv", log_amt=True):
    df = pd.read_csv(load_path)

    if Path(save_path).exists():
        print(f"File {save_path} already exists")
        return pd.read_csv(save_path)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    print("Start preprocessing...")

    assert not df.isna().any().any()
    print("Encode categorical...")
    cat_columns = [
        "trx_country", "mcc_code", "trans_type","card_cat_cd", 
        "ipt_name", "iso_crncy_cd", "ecom_fl","trx_direction"
    ]
    for col in cat_columns:
        frequency = df[col].value_counts().index
        frequency = pd.Series(np.arange(frequency.shape[0]), index = frequency)
        df[col] = df[col].map(frequency)
        frequency.to_csv(Path(save_path).parent / f"encoding_{col}.csv")

    if log_amt:
        print("Encode amt...")
        df["amt"] = np.sign(df["amt"]) * np.log(df["amt"].abs() + 1)

    df["index"] = (df["epk_id"].astype("str") + "_" + df["report_date"])
    df = df.set_index("index")

    print("Transform time...")
    for col in ["report_date", "evt_dttm"]:
        df[col] = pd.to_datetime(df[col]).apply(lambda x: x.timestamp())
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    df["event_time"] = df["evt_dttm"]

    select_cols = [col for col in df if col not in ["evt_dttm", "event_time"]]
    df = df[["event_time"] + select_cols]
    print("Saving...")
    df.to_csv(save_path)
    print("Done")
    return df

def create_parquet(cfg_path, csv_path):
    cfg = read_yaml(cfg_path)
    if (Path(cfg["data_path"]).parent/"train.parquet").exists():
        print("Parquet already exists.")
        return 
    print("Creating parquet...")
    cfg["data_path"] = csv_path
    dataset = SequenceDataset(cfg)
    dataset.load_dataset()
    
if __name__ == "__main__":
    # Generation
    df = make_df(47, "data/synthetic/data.csv")  # 47000 to get real scale
    print(len(df))  # ~41000000 for sber data

    # Preprocessing
    df = preprocess("data/synthetic/data.csv", "data/synthetic/processed/data.csv", log_amt=True)
    print(df)

    # Parquet generation
    create_parquet("configs/synthetic.yaml", "data/synthetic/processed/data.csv")

    print(next(read_pyarrow_file("data/synthetic/processed/train.parquet")))
    print(next(read_pyarrow_file("data/synthetic/processed/test.parquet")))