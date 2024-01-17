from pathlib import Path
import numpy as np
import pandas as pd
import torch
import logging

import sys

sys.path.append("../")

import src.models.base_models
from src.data_load.dataloader import create_data_loaders, create_test_loader
from src.trainers.trainer_supervised import (
    AccuracyTrainerSupervised,
    AucTrainerSupervised,
    SimpleTrainerSupervised,
    MSETrainerSupervised,
)
from experiments.utils import get_parser, read_config
from experiments.pipeline import Pipeline
from functools import partialmethod
from tqdm import tqdm
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

class SupervisedPipeline(Pipeline):
    def _train_eval(self, run_name, data_conf, model_conf):
        """
        Returns metrics dict like {"train_acc": metric_value, "val_acc": metric_value}
        Make sure that metric_value is going to be MAXIMIZED (higher -> better)
        """
        ### Create loaders and train ###
        train_loader, valid_loader = create_data_loaders(data_conf, supervised=True)
        print(len(train_loader))
        another_test_loader = create_test_loader(data_conf)

        net = getattr(src.models.base_models, model_conf.model_name)(
            model_conf=model_conf, data_conf=data_conf
        )
        opt = torch.optim.Adam(
            net.parameters(), model_conf.lr, weight_decay=model_conf.weight_decay
        )

        if self.resume_list:
            seed = int(run_name.split('/')[-1].replace('seed_', ''))
            resume = self.resume_list[seed]
        else:
            resume = self.resume

        trainer = self.TrainerClass(
            model=net,
            optimizer=opt,
            train_loader=train_loader,
            val_loader=valid_loader,
            run_name=run_name,
            ckpt_dir=Path(self.log_dir).parent / "ckpt",
            ckpt_replace=True,
            ckpt_resume=resume,
            ckpt_track_metric=data_conf.track_metric,
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
        test_metric = trainer.test(another_test_loader)

        return {
            "train_metric": train_metric[data_conf.track_metric],
            "val_metric": val_metric[data_conf.track_metric],
            "test_metric": test_metric[data_conf.track_metric],
        }

    def _param_grid(self, trial, model_conf, data_conf):
        model_conf.classifier_gru_hidden_dim = trial.suggest_int(
            "classifier_gru_hidden_dim", 64, 800
        )
        return trial, model_conf, data_conf


def get_trainer_class(data_conf) -> type:
    logger = logging.getLogger("event_seq")
    metric = None
    trainer_types = {
        "accuracy": AccuracyTrainerSupervised,
        "roc_auc": AucTrainerSupervised,
        "mse": MSETrainerSupervised,
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


class GenSupervisedPipeline(Pipeline):
    def __init__(
        self,
        run_name,
        device,
        total_epochs,
        data_conf,
        model_conf,
        TrainerClass,
        resume,
        log_dir,
        valid_supervised_loader,
        console_lvl="warning",
        file_lvl="info",
    ):
        super().__init__(
            run_name,
            device,
            total_epochs,
            data_conf,
            model_conf,
            TrainerClass,
            resume,
            log_dir,
            console_lvl,
            file_lvl,
        )
        self.valid_supervised_loader = valid_supervised_loader

    def _train_eval(self, run_name, data_conf, model_conf):
        """
        Returns metrics dict like {"train_acc": metric_value, "val_acc": metric_value}
        Make sure that metric_value is going to be MAXIMIZED (higher -> better)
        """
        ### Create loaders and train ###
        train_loader, valid_loader = create_data_loaders(data_conf)
        another_test_loader = create_test_loader(data_conf)

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
            val_loader=self.valid_supervised_loader,
            run_name=run_name,
            ckpt_dir=Path(self.log_dir).parent / "ckpt_test",
            ckpt_replace=True,
            ckpt_resume=self.resume,
            ckpt_track_metric=data_conf.track_metric,
            metrics_on_train=False,
            total_epochs=self.total_epochs,
            device=self.device,
            model_conf=model_conf,
        )

        ### RUN TRAINING ###
        trainer.run()

        trainer.load_best_model()
        train_metric = trainer.test(train_loader)
        val_metric = trainer.test(self.valid_supervised_loader)
        test_metric = trainer.test(another_test_loader)

        results = {
            "train_metric": train_metric[data_conf.track_metric],
            "val_metric": val_metric[data_conf.track_metric],
            "test_metric": test_metric[data_conf.track_metric],
        }

        if data_conf.get("post_gen_FT", False):
            print("in")
            resume_path = trainer.best_checkpoint()
            res_ft = self.run_finetuning(
                run_name, resume_path, self.valid_supervised_loader, trainer._ckpt_dir
            )
            results.update(res_ft)

        return results

    def _param_grid(self, trial, model_conf, data_conf):
        model_conf.classifier_gru_hidden_dim = trial.suggest_int(
            "classifier_gru_hidden_dim", 64, 800
        )
        return trial, model_conf, data_conf

    def run_finetuning(self, run_name, resume_path, valid_supervised_loader, ckpt_dir):
        res_ft = {}
        self.data_conf.post_gen_FT = False
        for observed_real_data_num in self.data_conf["FT_number_objects"]:
            subset_path = self.create_subset(ckpt_dir, run_name, observed_real_data_num)
            self.data_conf.train_path = subset_path
            self.data_conf.valid_size = 0.0

            log_dir = self.log_dir
            trainer_class = self.TrainerClass
            super_pipe = GenSupervisedPipeline(
                run_name=f"{run_name}_FT_{observed_real_data_num}",
                device=self.device,
                total_epochs=self.data_conf.post_gen_FT_epochs,
                data_conf=self.data_conf,
                model_conf=self.model_conf,
                TrainerClass=trainer_class,
                resume=resume_path,
                log_dir=log_dir,
                console_lvl=self.console_lvl,
                file_lvl=self.file_lvl,
                valid_supervised_loader=valid_supervised_loader,
            )

            results = super_pipe.run_experiment(
                run_name=f"{run_name}_FT_{observed_real_data_num}",
                conf=self.data_conf,
                model_conf=self.model_conf,
                seed=0,
            )

            for k, v in results.items():
                res_ft[f"{k}_FT_{observed_real_data_num}"] = v

            super_pipe = GenSupervisedPipeline(
                run_name=f"{run_name}_noFT_{observed_real_data_num}",
                device=self.device,
                total_epochs=self.data_conf.post_gen_FT_epochs,
                data_conf=self.data_conf,
                model_conf=self.model_conf,
                TrainerClass=trainer_class,
                resume=None,
                log_dir=log_dir,
                console_lvl=self.console_lvl,
                file_lvl=self.file_lvl,
                valid_supervised_loader=valid_supervised_loader,
            )

            results = super_pipe.run_experiment(
                run_name=f"{run_name}_noFT_{observed_real_data_num}",
                conf=self.data_conf,
                model_conf=self.model_conf,
                seed=0,
            )

            for k, v in results.items():
                res_ft[f"{k}_no-FT_{observed_real_data_num}"] = v

        self.data_conf.post_gen_FT = True

        return res_ft

    def create_subset(self, ckpt_dir, run_name, observed_real_data_num):
        train = pd.read_parquet(self.data_conf.FT_train_path)
        subset = train.head(observed_real_data_num)
        subset_path = ckpt_dir / f"{run_name}" / "subset.parquet"
        subset.to_parquet(subset_path)
        return subset_path


if __name__ == "__main__":
    args = get_parser().parse_args()
    ### TRAINING SETUP ###
    data_conf = read_config(args.data_conf, "data_configs")
    model_conf = read_config(args.model_conf, "model_configs")
    TrainerClass = get_trainer_class(data_conf)
    pipeline = SupervisedPipeline(
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
        resume_list=args.resume_list
    )
    request = {"classifier_gru_hidden_dim": 800}
    #metrics = pipeline.run_experiment()
    metrics = pipeline.do_n_runs()
    # metrics = pipeline.optuna_setup(
    #     "val_metric",
    #     request_list=[request],
    #     n_startup_trials=2,
    #     n_trials=3,
    # )
    print(metrics)
