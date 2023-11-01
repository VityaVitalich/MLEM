import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pipeline import Pipeline
from utils import parse_args

sys.path.append("../")

import src.models.base_models
from configs.data_configs.contrastive.age import data_configs as age_data
from configs.data_configs.contrastive.physionet import data_configs as physionet_data
from configs.data_configs.contrastive.rosbank import data_configs as rosbank_data
from configs.data_configs.contrastive.taobao import data_configs as taobao_data
from configs.model_configs.contrastive.age import model_configs as age_model
from configs.model_configs.contrastive.physionet import model_configs as physionet_model
from configs.model_configs.contrastive.rosbank import model_configs as rosbank_model
from configs.model_configs.contrastive.taobao import model_configs as taobao_model
from src.data_load.dataloader import create_data_loaders, create_test_loader
from src.trainers.trainer_contrastive import (
    AccuracyTrainerContrastive,
    AucTrainerContrastive,
)


class ContrastivePipeline(Pipeline):
    def _train_eval(self, run_name, conf, model_conf):
        """
        Returns metrics dict like {"train_acc": metric_value, "val_acc": metric_value}
        Make sure that metric_value is going to be MAXIMIZED (higher -> better)
        """
        ### Create loaders and train ###
        train_loader, valid_loader, _ = create_data_loaders(
            conf, supervised=False, pinch_test=True
        )
        another_test_loader = create_test_loader(conf)

        conf.train.split_strategy = {"split_strategy": "NoSplit"}
        conf.val.split_strategy = {"split_strategy": "NoSplit"}
        (
            train_supervised_loader,
            valid_supervised_loader,
            test_supervised_loader,
        ) = create_data_loaders(conf, pinch_test=True)

        net = getattr(src.models.base_models, model_conf.model_name)(
            model_conf=model_conf, data_conf=conf
        )
        opt = torch.optim.Adam(
            net.parameters(), model_conf.lr, weight_decay=model_conf.weight_decay
        )
        trainer = TrainerClass(
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
            cv=False,
        )
        assert len(train_metric) == 1
        return {
            "train_metric": train_metric,
            "val_metric": val_metric,
            "test_metric": test_metric,
            "another_test_metric": another_test_metric,
        }

    def _param_grid(self, trial, model_conf, conf):
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
                "num_heads_enc", [1, 2, 4]
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
        return trial, model_conf, conf


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
        "taobao": AucTrainerContrastive,
        "age": AccuracyTrainerContrastive,
        "rosbank": AucTrainerContrastive,
        "physionet": AucTrainerContrastive,
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

    pipeline = ContrastivePipeline(
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

    request = {
        "features_emb_dim": 32,
        "classifier_gru_hidden_dim": 128,
        "encoder": "TransformerEncoder",
        "num_enc_layers": 2,
        "use_numeric_emb": True,
        "loss.projector": "Linear",
        "loss.project_dim": 256,
        "num_heads_enc": 4,
        "loss.loss_fn": "ContrastiveLoss",
        "encoder_norm": "LayerNorm",
    }
    pipeline.optuna_setup(
        "val_metric",
        request_list=[request],
        n_startup_trials=0,
        n_trials=1,
        n_runs=3,
    )