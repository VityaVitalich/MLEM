from argparse import ArgumentParser

import optuna
import pandas as pd


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--run-name", help="run name for Trainer", default=None)
    parser.add_argument(
        "--console-log",
        help="console log level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="warning",
    )
    parser.add_argument(
        "--file-log",
        help="file log level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
    )
    parser.add_argument("--device", help="torch device to run on", default="cuda")
    parser.add_argument(
        "--log-dir",
        help="directory to write log file to",
        default="./logs",
    )
    parser.add_argument(
        "--total-epochs",
        help="total number of epochs to train",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--resume",
        help="path to checkpoint to resume from",
        default=None,
    )
    parser.add_argument(
        "--n-runs",
        help="number of runs with different seed",
        default=3,
        type=int,
    )
    parser.add_argument("--dataset", help="dataset", type=str, default="physionet")
    args = parser.parse_args()
    return args

def optuna_df(name="age/logs/optuna_ContrastiveLoss"):
    study = optuna.load_study(study_name=name, storage="sqlite:///example.db")

    # Retrieve data from the study and construct a pandas DataFrame
    data = []

    for trial in study.trials:
        trial_data = {
            "number": trial.number,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
            "value": trial.value,
            "datetime_start": trial.datetime_start,
            "datetime_complete": trial.datetime_complete
        }
        data.append(trial_data)

    df = pd.DataFrame(data)
    df["time"] = df["datetime_complete"] - df["datetime_start"]
    for k in df["user_attrs"].iloc[0]:
        df[k] = [i.get(k, "NaT") for i in df["user_attrs"]]
        params = pd.DataFrame([r for r in df["params"]], index=df.index)
    df = pd.concat([df, params], axis=1)
    df = df.drop(columns=["params", "datetime_start", "datetime_complete", "number", "user_attrs"])
    
    return df