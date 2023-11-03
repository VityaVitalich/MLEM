from pathlib import Path
import numpy as np
import pandas as pd
import torch
import logging
import pickle

import sys 
sys.path.append("../..")

import src.models.base_models
from src.data_load.dataloader import create_data_loaders, create_test_loader
from src.trainers.trainer_contrastive import (
    AccuracyTrainerContrastive,
    AucTrainerContrastive,
    SimpleTrainerContrastive,
)
from experiments.utils import get_parser, read_config
from experiments.pipeline import Pipeline
from experiments.pipeline_contrastive import get_trainer_class


class SberPipeline(Pipeline):
    def _train_eval(self, run_name, data_conf, model_conf):
        """
        Returns metrics dict like {"train_acc": metric_value, "val_acc": metric_value}
        Make sure that metric_value is going to be MAXIMIZED (higher -> better)
        """
        ### Create loaders and train ###
        train_loader, valid_loader = create_data_loaders(data_conf, supervised=False)
        # another_test_loader = create_test_loader(data_conf)

        # data_conf.train.split_strategy = {"split_strategy": "NoSplit"}
        # data_conf.val.split_strategy = {"split_strategy": "NoSplit"}
        # (
        #     train_supervised_loader,
        #     valid_supervised_loader,
        #     test_supervised_loader,
        # ) = create_data_loaders(data_conf, pinch_test=True)

        net = getattr(src.models.base_models, model_conf.model_name)(
            model_conf=model_conf, data_conf=data_conf
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
            ckpt_track_metric="total_loss", # "epoch" TODO why?
            metrics_on_train=False,
            total_epochs=self.total_epochs,
            device=self.device,
            model_conf=model_conf,
        )

        ckpt_path = Path(self.log_dir).parent / "ckpt" / run_name
        with open(ckpt_path / "model_config.pkl", "wb") as f:
            pickle.dump(model_conf, f)
        with open(ckpt_path / "data_config.pkl", "wb") as f:
            pickle.dump(data_conf, f)

        ### RUN TRAINING ###
        trainer.run()

        # train_metric, (val_metric, test_metric, another_test_metric) = trainer.test(
        #     train_supervised_loader,
        #     (valid_supervised_loader, test_supervised_loader, another_test_loader),
        # )
        return {
            "val_metric": 0
            # "train_metric": train_metric,
            # "val_metric": val_metric,
            # "test_metric": test_metric,
            # "another_test_metric": another_test_metric,
        }

if __name__ == "__main__":
    args = get_parser().parse_args()
    ### TRAINING SETUP ###
    data_conf = read_config(args.data_conf, "data_configs")
    model_conf = read_config(args.model_conf, "model_configs")
    TrainerClass = get_trainer_class(data_conf)
    pipeline = SberPipeline(
        run_name=args.run_name,
        device=args.device,
        total_epochs=args.total_epochs,
        data_conf=data_conf,
        model_conf=model_conf,
        TrainerClass=TrainerClass,
        resume=args.resume,
        log_dir=args.log_dir,
        console_lvl=args.console_lvl,
        file_lvl=args.file_lvl,
    )
    request = {
    }
    metrics = pipeline.run_experiment()
    # metrics = pipeline.do_n_runs()
    # metrics = pipeline.optuna_setup(
    #     "val_metric",
    #     request_list=[request],
    #     n_startup_trials=2,
    #     n_trials=3,
    # )
    print(metrics)