from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import src.models.base_models
from src.data_load.dataloader import create_data_loaders, create_test_loader
from src.trainers.trainer_gen import GenTrainer, GANGenTrainer
import src.models.gen_models
from experiments.utils import parse_args, setup_logging, read_config
from experiments.pipeline_supervised import (
    SupervisedPipeline, 
    get_trainer_class as get_supervised_trainer_class,
)
# from configs.model_configs.gen.age_genval import (
#     model_configs as model_configs_genval,
# )
# from configs.model_configs.gen.age_D import (
#     model_configs as model_configs_D,
# )
# Dont spam new condigs!!!!!! TODO
# model_conf.D = model_configs_D
# model_conf.genval = model_configs_genval

class GenerativePipeline(Pipeline):
    def _train_eval(self, run_name, data_conf, model_conf):
        """
        Returns metrics dict like {"train_acc": metric_value, "val_acc": metric_value}
        Make sure that metric_value is going to be MAXIMIZED (higher -> better)
        """
        ### Create loaders and train ###
        train_loader, valid_loader, _ = create_data_loaders(data_conf, supervised=False, pinch_test=True)
        another_test_loader = create_test_loader(data_conf)
        # conf.valid_size = 0
        data_conf.train.split_strategy = {"split_strategy": "NoSplit"}
        data_conf.val.split_strategy = {"split_strategy": "NoSplit"}
        train_supervised_loader, valid_supervised_loader, test_loader = create_data_loaders(data_conf, pinch_test=True)

        net = getattr(src.models.gen_models, model_conf.model_name)(
            model_conf=model_conf, data_conf=data_conf
        )
        opt = torch.optim.Adam(
            net.parameters(), model_conf.lr, weight_decay=model_conf.weight_decay
        )
        if model_conf.use_discriminator:
            model_conf_D = model_conf.D
            D = getattr(src.models.base_models, model_conf_D.model_name)(
                model_conf=model_conf, data_conf=data_conf
            )
            D_opt = torch.optim.Adam(
                D.parameters(), model_conf_D.lr, weight_decay=model_conf_D.weight_decay
            )

            trainer = GANGenTrainer(
                model=net,
                optimizer=opt,
                discriminator=D,
                d_opt=D_opt,
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
                model_conf_d=model_conf_D,
            )
        else:
            trainer = GenTrainer(
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

        ### RUN TRAINING ###
        trainer.run()

        # trainer.load_best_model()
        trainer.test(test_loader, train_supervised_loader) # TODO return metrics

        if self.gen_val: # TODO прокинуть в init
            generated_data_path = trainer.generate_data(train_supervised_loader)
            data_conf.train_supervised_path = generated_data_path
            data_conf.valid_size = 0.1

            run_name = run_name # TODO check that there is no logging colision. Maybe go to subdir. E.g. run_name=run_name/"sup"
            total_epochs = self.gen_val_epoch # TODO прокинуть в init
            model_conf_genval = model_conf.genval
            log_dir = "./logs/generations/"
            trainer_class = self.TrainerClass
            super_pipe = SupervisedPipeline(
                run_name=run_name,
                device=self.device,
                total_epochs=total_epochs,
                data_conf=data_conf,
                model_conf=model_conf_genval,
                TrainerClass=trainer_class,
                resume=None,
                log_dir=log_dir,
                file_log=self.file_log,
            )
            summary_df = super_pipe.do_n_runs(
                n_runs=3
            )
            # logger.info(f"Generated test metric: {generated_test_metric};") TODO return metric


if __name__ == "__main__":
    args = parse_args()
    parser = ArgumentParser()
    parser.add_argument(
        "--gen-val",
        help="Whether to perform generated validation",
        default=False,
    )
    parser.add_argument(
        "--gen-val-epoch",
        help="How many epochs to perform on generated samples",
        default=15,
        type=int,
    )
    new_args = parser.parse_args()
    args.gen_val = new_args.gen_val
    args.gen_val_epoch = new_args.gen_val_epoch

    setup_logging(args.console_log)

    ## TRAINING SETUP ###
    data_conf = read_config(args.data_conf, "data_configs")
    model_conf = read_config(args.model_conf, "model_configs")
    TrainerClass = get_supervised_trainer_class(data_conf)
    log_dir = args.log_dir

    pipeline = GenerativePipeline(
        run_name=args.run_name,
        device=args.device,
        total_epochs=args.total_epochs,
        data_conf=data_conf,
        model_conf=model_conf,
        TrainerClass=TrainerClass,
        resume=args.resume,
        log_dir=log_dir,
        console_log=args.console_log,
        file_log=args.file_log,
    )
