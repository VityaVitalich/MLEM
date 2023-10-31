from argparse import ArgumentParser

import pandas as pd
from pathlib import Path
from ml_collections import ConfigDict
import logging
from contextlib import contextmanager
from typing import Union


@contextmanager
def log_to_file(filename: Path, log_level: Union[str, int]):
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())

    fh = logging.FileHandler(filename)
    fh.setLevel(log_level)
    ffmt = logging.Formatter(
        "{levelname:8} - {process: ^6} - {name: ^16} - {asctime} - {message}",
        style="{",
    )
    fh.setFormatter(ffmt)
    logger = logging.getLogger("event_seq")
    logger.addHandler(fh)
    lvl = logger.getEffectiveLevel()
    logger.setLevel(min(lvl, log_level))

    try:
        yield
    finally:
        fh.close()
        logger.removeHandler(fh)


def setup_logging(cons_lvl: Union[str, int] = "warning"):
    if isinstance(cons_lvl, str):
        cons_lvl = getattr(logging, cons_lvl.upper())

    ch = logging.StreamHandler()
    ch.setLevel(cons_lvl)
    cfmt = logging.Formatter("{levelname:8} - {asctime} - {message}", style="{")
    ch.setFormatter(cfmt)

    logger = logging.getLogger("event_seq")
    logger.setLevel(cons_lvl)
    logger.addHandler(ch)


def read_config(conf_path: Union[Path, str], func_name: str) -> ConfigDict:
    if isinstance(conf_path, str):
        conf_path = Path(conf_path)

    source = conf_path.read_text()
    bytecode = compile(source, conf_path.as_posix(), "exec")
    namespace = {
        "__file__": conf_path.as_posix(),
    }
    exec(bytecode, namespace)
    return namespace[func_name]()  # type: ignore


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--run-name", help="run name for Trainer", default=None)
    parser.add_argument("--data-conf", help="path to data config", required=True)
    parser.add_argument("--model-conf", help="path to model config", required=True)
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
        default="./experiments/logs",
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
    args = parser.parse_args()
    return args

def optuna_df(name="age/logs/optuna_ContrastiveLoss"):
    import optuna

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

if __name__ == "__main__":
    args = parse_args()
