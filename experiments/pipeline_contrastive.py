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
)
from experiments.utils import get_parser, read_config
from experiments.pipeline import Pipeline


class ContrastivePipeline(Pipeline):
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
            ckpt_dir=Path(self.log_dir).parent / "ckpt",
            ckpt_replace=True,
            ckpt_resume=self.resume,
            ckpt_track_metric="epoch",
            metrics_on_train=False,
            total_epochs=self.total_epochs,
            device=self.device,
            model_conf=model_conf,
        )

        ### RUN TRAINING ###
        trainer.run()

        train_metric, (val_metric, test_metric, another_test_metric) = trainer.test(
            train_supervised_loader,
            (valid_supervised_loader, test_supervised_loader, another_test_loader),
        )
        return {
            "train_metric": train_metric,
            "val_metric": val_metric,
            "test_metric": test_metric,
            "another_test_metric": another_test_metric,
        }

    def _param_grid(self, trial, model_conf, data_conf):
        for param in (
            ("batch_first_encoder", [True, False]),
            ("features_emb_dim", [4, 8, 16, 32]),
            ("classifier_gru_hidden_dim", [16, 32, 64, 128]),
            ("encoder", ["Identity", "TransformerEncoder"]),
            ("num_enc_layers", [1, 2]),
            ("after_enc_dropout", [0.0, 0.1, 0.2]),
            ("activation", ["ReLU", "LeakyReLU", "Mish", "Tanh"]),
            ("lr", [3e-4, 1e-3, 3e-3]),
            ("weight_decay", [1e-5, 1e-4, 1e-3]),
            ("use_numeric_emb", [True, False]),
        ):
            model_conf[param[0]] = trial.suggest_categorical(param[0], param[1])
        if model_conf["use_numeric_emb"]:
            model_conf["numeric_emb_size"] = model_conf["features_emb_dim"]
            model_conf["num_heads_enc"] = trial.suggest_categorical(
                "num_heads_enc", [1]  # TODO complicated not to fail
            )
        else:
            model_conf["num_heads_enc"] = 1

        if model_conf["encoder"] == "TransformerEncoder":
            model_conf["encoder_norm"] = trial.suggest_categorical(
                "encoder_norm", ["Identity", "LayerNorm"]
            )  # important to know corr to encoder
        elif model_conf["encoder"] == "Identity":
            model_conf["encoder_norm"] = "Identity"

        for param in (
            (
                "loss.loss_fn",
                ["ContrastiveLoss"],
            ),  # , "InfoNCELoss", "DecoupledInfoNCELoss", "RINCELoss", "DecoupledPairwiseInfoNCELoss"]),
            ("loss.projector", ["Identity", "Linear", "MLP"]),
            ("loss.project_dim", [32, 64, 128, 256]),
        ):
            model_conf.loss[param[0].split(".")[1]] = trial.suggest_categorical(
                param[0], param[1]
            )
        if model_conf.loss.loss_fn == "ContrastiveLoss":
            model_conf.loss.margin = trial.suggest_categorical(
                "loss.margin", [0.0, 0.1, 0.3, 0.5, 1.0]
            )
        else:
            model_conf.loss.temperature = trial.suggest_categorical(
                "loss.temperature", [0.01, 0.03, 0.1, 0.3, 1.0]
            )
        if model_conf.loss.loss_fn == "InfoNCELoss":
            model_conf.loss.angular_margin = trial.suggest_categorical(
                "loss.angular_margin", [0.0, 0.3, 0.5, 0.7]
            )
        elif model_conf.loss.loss_fn == "RINCELoss":
            model_conf.loss.q = trial.suggest_categorical(
                "loss.q", [0.01, 0.03, 0.1, 0.3]
            )
            model_conf.loss.lam = trial.suggest_categorical(
                "loss.lam", [0.003, 0.01, 0.03, 0.1, 0.3]
            )
        return trial, model_conf, data_conf


def get_trainer_class(data_conf) -> type:
    logger = logging.getLogger("event_seq")
    metric = None
    trainer_types = {
        "accuracy": AccuracyTrainerContrastive,
        "roc_auc": AucTrainerContrastive,
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
    args = get_parser().parse_args()
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
    )
    request = {
        "features_emb_dim": 32,
        "classifier_gru_hidden_dim": 128,
        "encoder": "TransformerEncoder",
        "num_enc_layers": 2,
        "use_numeric_emb": True,
        "loss.projector": "Linear",
        "loss.project_dim": 256,
        "num_heads_enc": 1,
        "loss.loss_fn": "ContrastiveLoss",
        "encoder_norm": "LayerNorm",
    }
    # metrics = pipeline.run_experiment()
    # metrics = pipeline.do_n_runs()
    metrics = pipeline.optuna_setup(
        "val_metric",
        request_list=[request],
        n_startup_trials=2,
        n_trials=3,
    )
    print(metrics)
