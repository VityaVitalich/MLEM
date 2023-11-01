import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from .pipeline import Pipeline
from .utils import parse_args

sys.path.append("../")

import src.models.base_models
from configs.data_configs.age import data_configs as age_data
from configs.data_configs.physionet import data_configs as physionet_data
from configs.data_configs.rosbank import data_configs as rosbank_data
from configs.data_configs.taobao import data_configs as taobao_data
from configs.model_configs.supervised.age import model_configs as age_model
from configs.model_configs.supervised.physionet import model_configs as physionet_model
from configs.model_configs.supervised.rosbank import model_configs as rosbank_model
from configs.model_configs.supervised.taobao import model_configs as taobao_model
from src.data_load.dataloader import create_data_loaders, create_test_loader
from src.trainers.trainer_supervised import (
    AccuracyTrainerSupervised,
    AucTrainerSupervised,
)


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


def get_data_config(dataset):
    config_dict = {
        "taobao": taobao_data,
        "age": age_data,
        "rosbank": rosbank_data,
        "physionet": physionet_data,
    }
    return config_dict[dataset]()


def get_model_config(dataset):
    config_dict = {
        "taobao": taobao_model,
        "age": age_model,
        "rosbank": rosbank_model,
        "physionet": physionet_model,
    }
    return config_dict[dataset]()


def get_trainer_class(dataset):
    trainer_dict = {
        "taobao": AucTrainerSupervised,
        "age": AccuracyTrainerSupervised,
        "rosbank": AucTrainerSupervised,
        "physionet": AucTrainerSupervised,
    }
    return trainer_dict[dataset]


if __name__ == "__main__":
    args = parse_args()

    ### TRAINING SETUP ###
    dataset = args.dataset
    conf = get_data_config(dataset)
    model_conf = get_model_config(dataset)
    TrainerClass = get_trainer_class(dataset)
    log_dir = Path(dataset) / args.log_dir

    pipeline = SupervisedPipeline(
        run_name=args.run_name,
        device=args.device,
        total_epochs=args.total_epochs,
        conf=conf,
        model_conf=model_conf,
        TrainerClass=TrainerClass,
        resume=args.resume,
        log_dir=log_dir,
        console_log=args.console_log,
        file_log=args.file_log,
    )
