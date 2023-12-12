from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import sys

sys.path.append("../")

import src.models.base_models
from src.data_load.dataloader import create_data_loaders, create_test_loader
from src.trainers.trainer_timegan import TGTrainer
from src.models.TimeGan import TG
from experiments.utils import get_parser, read_config, draw_generated
from experiments.pipeline import Pipeline
from experiments.pipeline_supervised import (
    GenSupervisedPipeline,
    SupervisedPipeline,
    get_trainer_class as get_supervised_trainer_class,
)


class GenerativePipeline(Pipeline):
    def __init__(
        self,
        run_name,
        device,
        total_epochs_recon,
        total_epochs_gen,
        total_epochs_joint,
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
        draw=False,
    ):
        super().__init__(
            run_name,
            device,
            total_epochs_joint,
            data_conf,
            model_conf,
            TrainerClass,
            resume,
            log_dir,
            console_lvl,
            file_lvl,
        )
        self.total_epochs_recon = total_epochs_recon
        self.total_epochs_gen = total_epochs_gen
        self.total_epochs_joint = total_epochs_joint
        self.gen_val = gen_val
        self.gen_val_epoch = gen_val_epoch
        self.recon_val = recon_val
        self.recon_val_epoch = recon_val_epoch
        self.draw = draw

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

        net = TG(model_conf=model_conf, data_conf=data_conf)
        opt_e_params = (
            list(net.encoder.parameters())
            + list(net.decoder.parameters())
            + list(net.processor.parameters())
            + list(net.embedding_predictor.parameters())
            + list(net.numeric_projector.parameters())
        )
        opt_e = torch.optim.Adam(
            opt_e_params, model_conf.lr, weight_decay=model_conf.weight_decay
        )

        opt_g_params = list(net.generator.parameters()) + list(
            net.supervisor.parameters()
        )
        opt_g = torch.optim.Adam(
            opt_g_params, model_conf.lr, weight_decay=model_conf.weight_decay
        )

        opt_d = torch.optim.Adam(
            net.discriminator.parameters(),
            model_conf.lr,
            weight_decay=model_conf.weight_decay,
        )

        trainer = TGTrainer(
            model=net,
            optimizer_e=opt_e,
            optimizer_d=opt_d,
            optimizer_g=opt_g,
            train_loader=train_loader,
            val_loader=valid_loader,
            run_name=run_name,
            ckpt_dir=Path(self.log_dir).parent / "ckpt",
            ckpt_replace=True,
            ckpt_resume=self.resume,
            ckpt_track_metric="total_loss",
            metrics_on_train=False,
            total_epochs_recon=self.total_epochs_recon,
            total_epochs_gen=self.total_epochs_gen,
            total_epochs_joint=self.total_epochs_joint,
            device=self.device,
            model_conf=model_conf,
            data_conf=data_conf,
        )

        ### RUN TRAINING ###
        trainer.run()

        # trainer.load_best_model()
        train_metric, (supervised_val_metric, supervised_test_metric, fixed_test_metric), lin_prob_test = trainer.test(
            train_supervised_loader,
            (valid_supervised_loader, test_supervised_loader, fixed_test_loader),
        )
        metrics = {
            "train_metric": train_metric,
            "val_metric": supervised_val_metric,
            "test_metric": supervised_test_metric,
            "other_test_metric": fixed_test_metric,
            "lin_prob_test": lin_prob_test,
        }

        true_train_path = data_conf.train_path
        if self.recon_val:
            reconstructed_data_path = trainer.reconstruct_data(train_supervised_loader)
            data_conf.train_path = reconstructed_data_path
            data_conf.valid_size = 0.1

            total_epochs = self.recon_val_epoch
            model_conf_genval = model_conf.genval
            log_dir = self.log_dir
            trainer_class = self.TrainerClass
            super_pipe = GenSupervisedPipeline(
                run_name=f"{run_name}/reconstruction",
                device=self.device,
                total_epochs=total_epochs,
                data_conf=data_conf,
                model_conf=model_conf_genval,
                TrainerClass=trainer_class,
                resume=None,
                log_dir=log_dir,
                console_lvl=self.console_lvl,
                file_lvl=self.file_lvl,
                valid_supervised_loader=valid_supervised_loader,
            )
            super_df = super_pipe.do_n_runs(
                n_runs=3, max_workers=3
            )  # if you short in gpu, change max_workers=1
            for k in super_df:
                metrics[f"reconstruction_mean_{k}"] = super_df.loc["mean", k]
        if self.gen_val:
            generated_data_path = trainer.generate_data(train_supervised_loader)
            data_conf.train_path = generated_data_path
            data_conf.valid_size = 0.1

            total_epochs = self.gen_val_epoch
            model_conf_genval = model_conf.genval
            log_dir = self.log_dir
            trainer_class = self.TrainerClass
            super_pipe = GenSupervisedPipeline(
                run_name=f"{run_name}/generation",
                device=self.device,
                total_epochs=total_epochs,
                data_conf=data_conf,
                model_conf=model_conf_genval,
                TrainerClass=trainer_class,
                resume=None,
                log_dir=log_dir,
                console_lvl=self.console_lvl,
                file_lvl=self.file_lvl,
                valid_supervised_loader=valid_supervised_loader,
            )
            super_df = super_pipe.do_n_runs(
                n_runs=3, max_workers=3
            )  # if you short in gpu, change max_workers=1
            for k in super_df:
                metrics[f"generation_mean_{k}"] = super_df.loc["mean", k]

        if self.draw:
            save_path = self.log_dir / run_name / "distributions.png"
            draw_generated(
                generated_path=generated_data_path,
                true_path=true_train_path,
                reconstructed_path=reconstructed_data_path,
                data_conf=self.data_conf,
                out_path=save_path,
            )
        return metrics

    def _param_grid(self, trial, model_conf, data_conf):
        model_conf.classifier_gru_hidden_dim = trial.suggest_int(
            "classifier_gru_hidden_dim", 16, 64
        )
        return trial, model_conf, data_conf


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument(
        "--total-epochs-recon",
        help="Number of reconstruction training",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--total-epochs-gen", help="Number of generative training", default=1, type=int
    )
    parser.add_argument(
        "--total-epochs-joint", help="Number of joint training", default=1, type=int
    )
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
    args = parser.parse_args()

    ## TRAINING SETUP ###
    data_conf = read_config(args.data_conf, "data_configs")
    model_conf = read_config(args.model_conf, "model_configs")
    TrainerClass = get_supervised_trainer_class(data_conf)
    pipeline = GenerativePipeline(
        run_name=args.run_name,
        device=args.device,
        total_epochs_recon=args.total_epochs_recon,
        total_epochs_gen=args.total_epochs_gen,
        total_epochs_joint=args.total_epochs_joint,
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
        draw=args.draw,
    )
    request = {"classifier_gru_hidden_dim": 16}
    metrics = pipeline.run_experiment()
    # metrics = pipeline.do_n_runs()
    # metrics = pipeline.optuna_setup(
    #     "val_metric",
    #     request_list=[request],
    #     n_startup_trials=2,
    #     n_trials=3,
    # )
    print(metrics)
