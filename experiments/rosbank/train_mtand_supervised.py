import logging
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import torch
import sys

sys.path.append("../../")

from configs.data_configs.rosbank import data_configs
from configs.model_configs.mTAN.rosbank import model_configs
from src.data_load.dataloader import create_data_loaders
from src.models.mTAND.model import MegaNetClassifier
from src.trainers.trainer_mTAND import MtandTrainerSupervised
from src.models.mTAND.base_models import SimpleClassifier

if __name__ == "__main__":
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
    parser.add_argument("--device", help="torch device to run on", default="cpu")
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
    args = parser.parse_args()

    run_name = args.run_name or "mtand"
    run_name += f"_{datetime.now():%F_%T}"

    ### SETUP LOGGING ###
    ch = logging.StreamHandler()
    cons_lvl = getattr(logging, args.console_log.upper())
    ch.setLevel(cons_lvl)
    cfmt = logging.Formatter("{levelname:8} - {asctime} - {message}", style="{")
    ch.setFormatter(cfmt)

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_dir) / f"{run_name}.log"
    fh = logging.FileHandler(log_file)
    file_lvl = getattr(logging, args.file_log.upper())
    fh.setLevel(file_lvl)
    ffmt = logging.Formatter(
        "{levelname:8} - {process: ^6} - {name: ^16} - {asctime} - {message}",
        style="{",
    )
    fh.setFormatter(ffmt)

    logger = logging.getLogger("event_seq")
    logger.setLevel(min(file_lvl, cons_lvl))
    logger.addHandler(ch)
    logger.addHandler(fh)

    ### TRAINING SETUP ###
    conf = data_configs()
    model_conf = model_configs()

    train_loader, valid_loader = create_data_loaders(conf)
    net = MegaNetClassifier(model_conf=model_conf, data_conf=conf)
    opt = torch.optim.Adam(
        net.parameters(), lr=model_conf.lr, weight_decay=model_conf.weight_decay
    )
    trainer = MtandTrainerSupervised(
        model=net,
        optimizer=opt,
        train_loader=train_loader,
        val_loader=valid_loader,
        run_name=run_name,
        ckpt_dir=Path(__file__).parent / "ckpt",
        ckpt_replace=True,
        ckpt_resume=args.resume,
        ckpt_track_metric="roc_auc",
        metrics_on_train=False,
        total_epochs=args.total_epochs,
        device=args.device,
    )

    ### RUN TRAINING ###
    trainer.run()
