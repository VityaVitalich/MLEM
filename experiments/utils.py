from argparse import ArgumentParser

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ml_collections import ConfigDict
import logging
from contextlib import contextmanager
from typing import Union
import sys


def draw_generated(generated_path, true_path, reconstructed_path, data_conf, out_path):
    if "num_plots" in data_conf.keys():
        draw_generated_pendulum(
            generated_path, true_path, reconstructed_path, data_conf, out_path
        )
    else:
        draw_generated_trx(
            generated_path, true_path, reconstructed_path, data_conf, out_path
        )


def draw_generated_pendulum(
    generated_path, true_path, reconstructed_path, data_conf, out_path
):
    train = pd.read_parquet(true_path)
    gen = pd.read_parquet(generated_path)
    recon = pd.read_parquet(reconstructed_path)

    train.index = np.arange(len(train))
    dataframes = [train, gen, recon]
    names = ["True", "Generated", "Reconstructed"]
    num_plots = data_conf["num_plots"]
    idx = np.random.choice(train.index)
    l = train["flag"][idx]

    start_idx = np.random.randint(0, train.loc[idx]["trx_count"] - num_plots, 1)[0]

    fig, axs = plt.subplots(num_plots, len(dataframes), figsize=(25, 25))

    for i, df in enumerate(dataframes):
        random_row = df.loc[idx]
        X = random_row[[str(i) for i in range(256)]].values
        images = np.vstack(X).T
        for j, idx in enumerate(range(start_idx, start_idx + num_plots)):
            image = images[idx]
            ax = axs[j, i]
            ax.imshow(image.reshape(16, 16), cmap="gray", vmin=0, vmax=1)

            time = random_row["event_time"][idx]
            ax.set_title(f"{names[i]} with time: {time}")

    # Adjust spacing between subplots
    plt.tight_layout()
    fig.suptitle(f"Length: {l}", fontsize=16, y=1.02)
    plt.savefig(out_path)


def draw_generated_trx(
    generated_path, true_path, reconstructed_path, data_conf, out_path
):
    train = pd.read_parquet(true_path)
    gen = pd.read_parquet(generated_path)
    recon = pd.read_parquet(reconstructed_path)

    cols = (
        data_conf.features.embeddings.keys() + data_conf.features.numeric_values.keys()
    )
    cols.append("event_time")

    dataframes = [train, gen, recon]
    names = ["True", "Generated", "Reconstructed"]
    num_plots = len(cols)
    fig, axs = plt.subplots(num_plots + 1, len(dataframes), figsize=(25, 25))

    # Iterate through dataframes and columns
    for i, df in enumerate(dataframes):
        for j, col in enumerate(cols):
            ax = axs[j, i]
            data = np.hstack(df[col].values)
            ax.hist(data, bins=100)
            ax.set_title(f"{names[i]} - {col}")
            ax.set_xlabel(col)

    # create bounded event time graph
    for i, df in enumerate(dataframes):
        ax = axs[j + 1, i]
        data = np.hstack(df["event_time"].values)

    # Adjust spacing between subplots
    plt.tight_layout()

    plt.savefig(out_path)
    # Show the plots
    # plt.show()


@contextmanager
def log_to_file(filename: Path, file_lvl="info", cons_lvl="warning"):
    if isinstance(file_lvl, str):
        file_lvl = getattr(logging, file_lvl.upper())
    if isinstance(cons_lvl, str):
        cons_lvl = getattr(logging, cons_lvl.upper())

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(cons_lvl)
    cfmt = logging.Formatter("{levelname:8} - {asctime} - {message}", style="{")
    ch.setFormatter(cfmt)

    fh = logging.FileHandler(filename)
    fh.setLevel(file_lvl)
    ffmt = logging.Formatter(
        "{levelname:8} - {process: ^6} - {name: ^16} - {asctime} - {message}",
        style="{",
    )
    fh.setFormatter(ffmt)
    logger = logging.getLogger("event_seq")
    logger.setLevel(min(file_lvl, cons_lvl))
    logger.addHandler(fh)
    logger.addHandler(ch)

    try:
        yield
    finally:
        fh.close()
        logger.removeHandler(fh)
        logger.removeHandler(ch)


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


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--run-name", help="run name for Trainer", default=None)
    parser.add_argument("--data-conf", help="path to data config", required=True)
    parser.add_argument("--model-conf", help="path to model config", required=True)
    parser.add_argument(
        "--console-lvl",
        help="console log level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="warning",
    )
    parser.add_argument(
        "--file-lvl",
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
    parser.add_argument(
        "--resume_list",
        action='append',
        help='list of checkpoints',
        default=None,
        type=str
    )
    print(parser)
    return parser


def optuna_df(name="age/logs/optuna_ContrastiveLoss"):
    import optuna
    from optuna.storages import JournalFileStorage, JournalStorage

    storage = JournalStorage(JournalFileStorage(f"{name}/study.log"))
    study = optuna.load_study(study_name=name, storage=storage)
    df = study.trials_dataframe()
    df = df.drop(
        columns=[
            "number",
        ]
    )

    return df, study


if __name__ == "__main__":
    args = get_parser().parse_args()
