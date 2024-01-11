from pathlib import Path
import numpy as np
import pandas as pd
import torch
import logging

import sys

sys.path.append("../")

import src.models.base_models
from src.data_load.dataloader import create_data_loaders, create_test_loader
from src.trainers.trainer_contrastive import (
    AccuracyTrainerContrastive,
    AucTrainerContrastive,
    SimpleTrainerContrastive,
    MSETrainerContrastive
)
from experiments.pipeline_supervised import (
    GenSupervisedPipeline,
    SupervisedPipeline,
    get_trainer_class as get_supervised_trainer_class,
)
import param_grids
from experiments.utils import get_parser, read_config
from experiments.pipeline import Pipeline

from tqdm import tqdm
from functools import partialmethod

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


class ContrastivePipeline(Pipeline):
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
        console_lvl="warning",
        file_lvl="info",
        FT_on_labeled=False,
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
        self.FT_on_labeled = FT_on_labeled
    def _train_eval(self, run_name, data_conf, model_conf):
        """
        Returns metrics dict like {"train_acc": metric_value, "val_acc": metric_value}
        Make sure that metric_value is going to be MAXIMIZED (higher -> better)
        """
        ### Create loaders and train ###
        train_loader, valid_loader, _ = create_data_loaders(
            data_conf, supervised=False, pinch_test=True
        )
        another_test_loader = create_test_loader(data_conf)

        data_conf.train.split_strategy = {"split_strategy": "NoSplit"}
        data_conf.val.split_strategy = {"split_strategy": "NoSplit"}
        (
            train_supervised_loader,
            valid_supervised_loader,
            test_supervised_loader,
        ) = create_data_loaders(data_conf, pinch_test=True)

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
            ckpt_dir=Path(self.log_dir) / run_name / "ckpt",
            ckpt_replace=True,
            ckpt_resume=self.resume,
            ckpt_track_metric="epoch",
            metrics_on_train=False,
            total_epochs=self.total_epochs,
            device=self.device,
            model_conf=model_conf,
            data_conf=data_conf,
        )

        ### RUN TRAINING ###
        trainer.run()
        trainer.load_best_model()

        (
            train_metric,
            (val_metric, test_metric, another_test_metric),
            train_logist,
            (val_logist, test_logist, another_test_logist),
            anisotropy,
            intrinsic_dimension,
        ) = trainer.test(
            train_supervised_loader,
            (valid_supervised_loader, test_supervised_loader, another_test_loader),
        )

        metrics = {
            "train_metric": train_metric,
            "val_metric": val_metric,
            "test_metric": test_metric,
            "another_test_metric": another_test_metric,
            "train_logist": train_logist,
            "val_logist": val_logist,
            "test_logist": test_logist,
            "another_test_logist": another_test_logist,
            "anisotropy": anisotropy,
            "intrinsic_dimension": intrinsic_dimension,
        }

        true_train_path = data_conf.train_path
        self.data_conf.FT_train_path = true_train_path

        if self.FT_on_labeled:
            resume_path = trainer.best_checkpoint()
            self.model_conf.model_name = "GRUClassifier"
            self.model_conf.predict_head = "Linear"
            if self.data_conf.track_metric == 'mse':
                self.model_conf.loss.loss_fn = "MSE"
            else:
                self.model_conf.loss.loss_fn = "CrossEntropy"

            res_ft = self.run_finetuning(run_name, resume_path, trainer._ckpt_dir)
            metrics.update(res_ft)

        return metrics

    def run_finetuning(self, run_name, resume_path, ckpt_dir):
        self.data_conf.train.split_strategy = {"split_strategy": "NoSplit"}
        self.data_conf.val.split_strategy = {"split_strategy": "NoSplit"}
        train = pd.read_parquet(self.data_conf.FT_train_path)
        train = train[~train[self.data_conf.features.target_col].isna()]

        ### Create Valid Loader to use in every run ###
        num_valid_rows = int(len(train) * 0.1)
        valid = train.tail(num_valid_rows)
        valid_path = ckpt_dir / f"{run_name}" / "valid_subset.parquet"
        valid.to_parquet(valid_path)
        data_conf = self.data_conf
        data_conf.valid_size = 0.0
        data_conf.train_path = valid_path
        valid_loader, _ = create_data_loaders(
            data_conf, supervised=True, pinch_test=False
        )
        ### Create Train path with only labeled samples and not including valid ###
        train_supervised = train.head(len(train) - num_valid_rows)
        train_supervised_path = ckpt_dir / f"{run_name}" / "train_sup_subset.parquet"
        train_supervised.to_parquet(train_supervised_path)
        self.data_conf.FT_train_path = train_supervised_path

        res_ft = {}
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
                valid_supervised_loader=valid_loader,
            )

            results = super_pipe.run_experiment(
                run_name=f"{run_name}_FT_{observed_real_data_num}",
                conf=self.data_conf,
                model_conf=self.model_conf,
                seed=0,
            )

            for k, v in results.items():
                res_ft[f"{k}_FT_{observed_real_data_num}"] = v

        return res_ft

    def create_subset(self, ckpt_dir, run_name, observed_real_data_num):
        train = pd.read_parquet(self.data_conf.FT_train_path)
        if observed_real_data_num == "all":
            observed_real_data_num = len(train)
        subset = train.head(observed_real_data_num)
        subset_path = ckpt_dir / f"{run_name}" / "subset.parquet"
        subset.to_parquet(subset_path)
        return subset_path

    def _param_grid(self, trial, model_conf, data_conf):
        return getattr(param_grids, self.grid_name)(trial, model_conf, data_conf)


def get_trainer_class(data_conf) -> type:
    logger = logging.getLogger("event_seq")
    metric = None
    trainer_types = {
        "accuracy": AccuracyTrainerContrastive,
        "roc_auc": AucTrainerContrastive,
        "mse": MSETrainerContrastive,
        None: SimpleTrainerContrastive,
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
    parser = get_parser()
    parser.add_argument(
        "--grid-name",
        help="Name of grid function from param_grids.py",
        default="",
        type=str,
    )
    parser.add_argument(
        "--FT",
        help="if to Fine-tune after Pre-Train",
        default=0,
        type=int,
    )
    args = parser.parse_args()
    ### TRAINING SETUP ###
    data_conf = read_config(args.data_conf, "data_configs")
    model_conf = read_config(args.model_conf, "model_configs")
    TrainerClass = get_trainer_class(data_conf)
    pipeline = ContrastivePipeline(
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
        FT_on_labeled=args.FT
    )
    request = [
        {
            "batch_first_encoder": True,
            "features_emb_dim": 16,
            "use_numeric_emb": False,
            "numeric_emb_size": 8,
            "encoder_feature_mixer": False,
            "classifier_gru_hidden_dim": 800,
            "use_deltas": False,
            "encoder": "Identity",
            "num_enc_layers": 1,
            "num_heads_enc": 1,
            "encoder_norm": "Identity",
            "after_enc_dropout": 0.0,
            "activation": "LeakyReLU",
            # "loss_fn": "ContrastiveLoss",
            "margin": 0.5,
            "projector": "Identity",
            "project_dim": 32,
            "lr": 0.001,
            "weight_decay": 0.0,
        },
    ]
   # metrics = pipeline.run_experiment()
    metrics = pipeline.do_n_runs()
    # metrics = pipeline.optuna_setup(
    #     "val_metric",
    #     request_list=request,
    #     n_startup_trials=3,
    #     n_trials=50,
    #     n_runs=3,
    # )
    print(metrics)
