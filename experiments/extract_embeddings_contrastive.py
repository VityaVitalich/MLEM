from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pickle
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
        trainer.load_ckpt(self.resume)
        ### RUN TRAINING ###
        preds, gts = trainer.predict(
            another_test_loader
        )
        save_path = Path(self.log_dir) / run_name / 'embeddings.pickle'
        gt_save_path = Path(self.log_dir) / run_name / 'gts.pickle'

        with open(save_path, 'wb') as f:
            pickle.dump(preds, f)
        with open(gt_save_path, 'wb') as f:
            pickle.dump(gts, f)
    

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
    metrics = pipeline.run_experiment()
    #metrics = pipeline.do_n_runs()
    # metrics = pipeline.optuna_setup(
    #     "val_metric",
    #     request_list=request,
    #     n_startup_trials=3,
    #     n_trials=50,
    #     n_runs=3,
    # )
    print(metrics)
