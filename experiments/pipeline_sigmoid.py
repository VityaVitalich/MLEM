from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import sys

sys.path.append("../")

import src.models.base_models
from src.data_load.dataloader import create_data_loaders, create_test_loader
from src.trainers.trainer_sigmoid import (
    SigmoidTrainer,
)
from src.trainers.trainer_gen import (
    GenTrainer,
    GANGenTrainer,
)
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
from src.trainers.randomness import seed_everything

class SigmoidPipeline(Pipeline):
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
        fixed_test_loader = create_test_loader(data_conf)

        data_conf.train.split_strategy = {"split_strategy": "NoSplit"}
        data_conf.val.split_strategy = {"split_strategy": "NoSplit"}
        (
            train_supervised_loader,
            valid_supervised_loader,
            test_supervised_loader,
        ) = create_data_loaders(data_conf, pinch_test=True)
        ### LOAD CONTRASTIVE CKPT ###
        contrastive_net = (
            getattr(src.models.base_models, model_conf.contrastive.model_name)(
                model_conf=model_conf.contrastive, data_conf=data_conf
            )
            .to(self.model_conf.device)
            .eval()
        )
        contrastive_ckpt = torch.load(data_conf.pre_trained_contrastive_path)
        contrastive_net.load_state_dict(contrastive_ckpt["model"], strict=True)
        print("Contrastive Net Loaded")
        seed_everything(
                data_conf.client_list_shuffle_seed,
                avoid_benchmark_noise=True,
                only_deterministic_algorithms=False,
        )

        net = getattr(src.models.gen_models, model_conf.model_name)(
            model_conf=model_conf, data_conf=data_conf
        )
        opt = torch.optim.Adam(
            net.parameters(), model_conf.lr, weight_decay=model_conf.weight_decay
        )
        trainer = SigmoidTrainer(
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
            contrastive_model=contrastive_net,
        )


        ### RUN TRAINING ###
        trainer.run()
        trainer.load_best_model()
        # trainer.load_best_model()
        (
            train_metric,
            (supervised_val_metric, supervised_test_metric, fixed_test_metric),
            (log_val_metric, log_test_metric, log_fixed_test_metric),
            intrinsic,
            anisotropy
        ) = trainer.test(
            train_supervised_loader,
            (valid_supervised_loader, test_supervised_loader, fixed_test_loader),
        )
        metrics = {
            "train_metric": train_metric,
            "val_metric": supervised_val_metric,
            "test_metric": supervised_test_metric,
            "other_metric": fixed_test_metric,
            "lin_prob_test": log_fixed_test_metric,
            "anisotropy": anisotropy,
            "intrinsic": intrinsic
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
        train = pd.read_parquet(self.data_conf.FT_train_path)
        train = train[~train[self.data_conf.features.target_col].isna()]

        ### Create Valid Loader to use in every run ###
        num_valid_rows = int(len(train) * 0.1)
        valid = train.tail(num_valid_rows)
        valid_path = ckpt_dir / f"{run_name}" / "valid_subset.parquet"
        valid.to_parquet(valid_path)
        self.data_conf.train.split_strategy = {"split_strategy": "NoSplit"}
        self.data_conf.val.split_strategy = {"split_strategy": "NoSplit"}
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
        model_conf.classifier_gru_hidden_dim = trial.suggest_int(
            "classifier_gru_hidden_dim", 16, 64
        )
        return trial, model_conf, data_conf


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
    pipeline = SigmoidPipeline(
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
        FT_on_labeled=args.FT,
    )
    request = {"classifier_gru_hidden_dim": 16}
  #  metrics = pipeline.run_experiment()
    metrics = pipeline.do_n_runs()
    # metrics = pipeline.optuna_setup(
    #     "val_metric",
    #     request_list=[request],
    #     n_startup_trials=2,
    #     n_trials=3,
    # )
    print(metrics)
