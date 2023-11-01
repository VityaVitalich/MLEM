from pathlib import Path
import numpy as np
import pandas as pd
import torch

import logging

import src.models.base_models
from src.data_load.dataloader import create_data_loaders, create_test_loader
from src.trainers.trainer_supervised import (
    AccuracyTrainerSupervised,
    AucTrainerSupervised,
    SimpleTrainerSupervised,
)
from experiments.utils import parse_args, read_config
from experiments.pipeline import Pipeline


class SupervisedPipeline(Pipeline):
    def _train_eval(self, run_name, conf, model_conf):
        """
        Returns metrics dict like {"train_acc": metric_value, "val_acc": metric_value}
        Make sure that metric_value is going to be MAXIMIZED (higher -> better)
        """
        ### Create loaders and train ###
        train_loader, valid_loader = create_data_loaders(conf)
        another_test_loader = create_test_loader(conf)

        net = getattr(src.models.base_models, model_conf.model_name)(
            model_conf=model_conf, data_conf=conf
        )
        opt = torch.optim.Adam(
            net.parameters(), model_conf.lr, weight_decay=model_conf.weight_decay
        )
        trainer = self.TrainerClass(
            model=net,
            optimizer=opt,
            train_loader=train_loader,
            val_loader=valid_loader,
            run_name=run_name,
            ckpt_dir=Path(self.log_dir).parent / "ckpt",
            ckpt_replace=True,
            ckpt_resume=self.resume,
            ckpt_track_metric=conf.track_metric,
            metrics_on_train=False,
            total_epochs=self.total_epochs,
            device=self.device,
            model_conf=model_conf,
        )

        ### RUN TRAINING ###
        trainer.run()

        trainer.load_best_model()
        train_metric = trainer.test(train_loader)
        val_metric = trainer.test(valid_loader)
        another_test_metric = trainer.test(another_test_loader)

        return {
            "train_metric": train_metric,
            "val_metric": val_metric,
            # "test_metric": test_metric,
            "another_test_metric": another_test_metric,
        }


def get_trainer_class(data_conf) -> type:
    logger = logging.getLogger("event_seq")
    metric = None
    trainer_types = {
        "accuracy": AccuracyTrainerSupervised,
        "roc_auc": AucTrainerSupervised,
        None: SimpleTrainerSupervised,
    }

    if hasattr(data_conf, "main_metric"):
        metric = data_conf.main_metric
    elif hasattr(data_conf, "track_metric"):
        logger.warning(
            "`main_metric` field is not set in data config. "
            "Picking apropriate trainer based on `track_metric` field."
        )
        metric = data_conf.track_metric
    else:
        logger.warning(
            "Neither the `main_metric`, nor the `track_metric` fields are specified"
            " in the data config. Falling back to the simple contrastive trainer."
        )

    try:
        return trainer_types[metric]
    except KeyError:
        raise ValueError(f"Unkown metric: {metric}")


if __name__ == "__main__":
    args = parse_args()

    ### TRAINING SETUP ###
    data_conf = read_config(args.data_conf, "data_configs")
    model_conf = read_config(args.model_conf, "model_configs")

    TrainerClass = get_trainer_class(data_conf)
    log_dir = args.log_dir

    pipeline = SupervisedPipeline(
        run_name=args.run_name,
        device=args.device,
        total_epochs=args.total_epochs,
        data_conf=data_conf,
        model_conf=model_conf,
        TrainerClass=TrainerClass,
        resume=args.resume,
        log_dir=log_dir,
        console_log=args.console_log,
        file_log=args.file_log,
    )
