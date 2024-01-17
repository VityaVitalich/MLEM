from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import torch

import sys

sys.path.append("../")

import src.models.base_models
from src.data_load.dataloader import create_data_loaders, create_test_loader
from src.trainers.trainer_gen import (
    GenTrainer,
    GANGenTrainer,
)
from src.trainers.trainer_sigmoid import (
    SigmoidTrainer,
)
from src.trainers.trainer_ddpm import TrainerDDPM
import src.models.gen_models
from experiments.utils import get_parser, read_config, draw_generated
from experiments.pipeline import Pipeline
from experiments.pipeline_supervised import (
    GenSupervisedPipeline,
    SupervisedPipeline,
    get_trainer_class as get_supervised_trainer_class,
)
from src.trainers.randomness import seed_everything

from functools import partialmethod
from tqdm import tqdm
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

class GenerativePipeline(Pipeline):
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
        gen_val,
        gen_val_epoch,
        recon_val,
        recon_val_epoch,
        console_lvl="warning",
        file_lvl="info",
        draw_generated=False,
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
        self.gen_val = gen_val
        self.gen_val_epoch = gen_val_epoch
        self.recon_val = recon_val
        self.recon_val_epoch = recon_val_epoch
        self.draw_generated = draw_generated
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
        fixed_test_loader = create_test_loader(data_conf)

        data_conf.train.split_strategy = {"split_strategy": "NoSplit"}
        data_conf.val.split_strategy = {"split_strategy": "NoSplit"}
        (
            train_supervised_loader,
            valid_supervised_loader,
            test_supervised_loader,
        ) = create_data_loaders(data_conf, pinch_test=True)

        net = getattr(src.models.gen_models, model_conf.model_name)(
            model_conf=model_conf, data_conf=data_conf
        )
        opt = torch.optim.Adam(
            net.parameters(), model_conf.lr, weight_decay=model_conf.weight_decay
        )
        gen_trainer_class = (
                TrainerDDPM if "DDPM" in self.model_conf.model_name else GenTrainer
        )
        trainer = gen_trainer_class(
            model=net,
            optimizer=opt,
            train_loader=train_loader,
            val_loader=valid_loader,
            run_name=run_name,
            ckpt_dir=Path(self.log_dir).parent / "ckpt",
            ckpt_replace=True,
            ckpt_resume=self.resume,
            ckpt_track_metric="total_loss",
            metrics_on_train=False,
            total_epochs=self.total_epochs,
            device=self.device,
            model_conf=model_conf,
            data_conf=data_conf,
        )
        trainer.load_ckpt(self.resume)

        preds, gts = trainer.get_embeddings(fixed_test_loader)
        save_path = Path(self.log_dir) / run_name / 'embeddings.pickle'
        gt_save_path = Path(self.log_dir) / run_name / 'gts.pickle'

        with open(save_path, 'wb') as f:
            pickle.dump(preds, f)
        with open(gt_save_path, 'wb') as f:
            pickle.dump(gts, f)
    



if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument(
        "--gen-val", help="Whether to perform generated validation", default=0, type=int
    )
    parser.add_argument(
        "--gen-val-epoch",
        help="How many epochs to perform on generated samples",
        default=15,
        type=int,
    )
    parser.add_argument(
        "--recon-val",
        help="Whether to perform generated validation",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--recon-val-epoch",
        help="How many epochs to perform on generated samples",
        default=25,
        type=int,
    )
    parser.add_argument(
        "--draw",
        help="if to draw distributions of gen and recon",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--FT",
        help="if to Fine-tune after Pre-Train",
        default=0,
        type=int,
    )
    args = parser.parse_args()

    print(args.recon_val)
    ## TRAINING SETUP ###
    data_conf = read_config(args.data_conf, "data_configs")
    model_conf = read_config(args.model_conf, "model_configs")
    TrainerClass = get_supervised_trainer_class(data_conf)
    print('trainer')
    pipeline = GenerativePipeline(
        run_name=args.run_name,
        device=args.device,
        total_epochs=args.total_epochs,
        data_conf=data_conf,
        model_conf=model_conf,
        TrainerClass=TrainerClass,
        resume=args.resume,
        log_dir=args.log_dir,
        gen_val=args.gen_val,
        gen_val_epoch=args.gen_val_epoch,
        recon_val=args.recon_val,
        recon_val_epoch=args.recon_val_epoch,
        console_lvl=args.console_lvl,
        file_lvl=args.file_lvl,
        draw_generated=args.draw,
        FT_on_labeled=args.FT,
    )
    request = {"classifier_gru_hidden_dim": 16}
    metrics = pipeline.run_experiment(seed=1)
    #metrics = pipeline.do_n_runs()
    # metrics = pipeline.optuna_setup(
    #     "val_metric",
    #     request_list=[request],
    #     n_startup_trials=2,
    #     n_trials=3,
    # )
    print(metrics)
