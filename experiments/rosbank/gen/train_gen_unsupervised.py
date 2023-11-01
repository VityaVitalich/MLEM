import logging
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import torch
import sys
import os

sys.path.append("../../../")

from configs.data_configs.rosbank import data_configs
from configs.model_configs.gen.rosbank import model_configs
from src.data_load.dataloader import create_data_loaders, create_test_loader
from src.trainers.trainer_gen import GenTrainer, GANGenTrainer
from src.trainers.randomness import seed_everything
import src.models.gen_models
import src.models.base_models
from copy import deepcopy

from experiments.pipeline_supervised import SupervisedPipeline, get_trainer_class
from configs.model_configs.gen.rosbank_genval import (
    model_configs as model_configs_genval,
)
from configs.model_configs.gen.rosbank_D import (
    model_configs as model_configs_D,
)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run-name", help="run name for Trainer", default=None)
    parser.add_argument(
        "--console-log",
        help="console log level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="warning",
    )
    parser.add_argument(
        "--file-log",
        help="file log level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
    )
    parser.add_argument("--device", help="torch device to run on", default="cpu")
    parser.add_argument(
        "--log-dir",
        help="directory to write log file to",
        default="./logs",
    )
    parser.add_argument(
        "--total-epochs",
        help="total number of epochs to train",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--resume",
        help="path to checkpoint to resume from",
        default=None,
    )
    parser.add_argument(
        "--recon-val",
        help="Whether to perform generated validation",
        default=False,
    )
    parser.add_argument(
        "--recon-val-epoch",
        help="How many epochs to perform on generated samples",
        default=25,
        type=int,
    )
    parser.add_argument(
        "--gen-val",
        help="Whether to perform generated validation",
        default=False,
    )
    parser.add_argument(
        "--gen-val-epoch",
        help="How many epochs to perform on generated samples",
        default=25,
        type=int,
    )
    args = parser.parse_args()

    run_name = args.run_name or "mtand"
    run_name += f"_{datetime.now():%F_%T}"

    ### SETUP LOGGING ###
    ch = logging.StreamHandler()
    cons_lvl = getattr(logging, args.console_log.upper())
    ch.setLevel(cons_lvl)
    cfmt = logging.Formatter("{levelname:8} - {asctime} - {message}", style="{")
    ch.setFormatter(cfmt)

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_dir) / f"{run_name}.log"
    fh = logging.FileHandler(log_file)
    file_lvl = getattr(logging, args.file_log.upper())
    fh.setLevel(file_lvl)
    ffmt = logging.Formatter(
        "{levelname:8} - {process: ^6} - {name: ^16} - {asctime} - {message}",
        style="{",
    )
    fh.setFormatter(ffmt)

    logger = logging.getLogger("event_seq")
    logger.setLevel(min(file_lvl, cons_lvl))
    logger.addHandler(ch)
    logger.addHandler(fh)

    ### TRAINING SETUP ###
    conf = data_configs()
    model_conf = model_configs()

    ### Fix randomness ###
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    seed_everything(
        conf.client_list_shuffle_seed,
        avoid_benchmark_noise=True,
        only_deterministic_algorithms=False,
    )

    ### Create loaders and train ###
    train_loader, valid_loader = create_data_loaders(conf, supervised=False)
    test_loader = create_test_loader(conf)
    conf.valid_size = 0
    conf.train.split_strategy = {"split_strategy": "NoSplit"}
    train_supervised_loader, _ = create_data_loaders(conf)

    model = getattr(src.models.gen_models, model_conf.model_name)
    net = model(model_conf=model_conf, data_conf=conf)
    opt = torch.optim.Adam(
        net.parameters(), model_conf.lr, weight_decay=model_conf.weight_decay
    )
    if model_conf.use_discriminator:
        model_conf_D = model_configs_D()
        D = getattr(src.models.base_models, model_conf_D.model_name)
        D = D(model_conf=model_conf_D, data_conf=conf)
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
            ckpt_dir=Path(__file__).parent / "ckpt",
            ckpt_replace=True,
            ckpt_resume=args.resume,
            ckpt_track_metric="total_loss",
            metrics_on_train=False,
            total_epochs=args.total_epochs,
            device=args.device,
            model_conf=model_conf,
            data_conf=conf,
            model_conf_d=model_conf_D,
        )

    else:
        trainer = GenTrainer(
            model=net,
            optimizer=opt,
            train_loader=train_loader,
            val_loader=valid_loader,
            run_name=run_name,
            ckpt_dir=Path(__file__).parent / "ckpt",
            ckpt_replace=True,
            ckpt_resume=args.resume,
            ckpt_track_metric="total_loss",
            metrics_on_train=False,
            total_epochs=args.total_epochs,
            device=args.device,
            model_conf=model_conf,
            data_conf=conf,
        )

    ### RUN TRAINING ###
    trainer.run()

    # trainer.load_best_model()
    trainer.test(test_loader, train_supervised_loader)

    if args.recon_val:
        reconstructed_data_path = trainer.reconstruct_data(train_supervised_loader)
        conf.train_path = reconstructed_data_path
        print(conf.train_path)
        conf.valid_size = 0.1

        run_name = run_name
        total_epochs = args.recon_val_epoch
        model_conf_genval = model_configs_genval()
        log_dir = "./logs/reconstructions/"
        recon_trainer_class = get_trainer_class("rosbank")
        recon_pipeline = SupervisedPipeline(
            run_name=run_name,
            device=args.device,
            total_epochs=total_epochs,
            conf=conf,
            model_conf=model_conf_genval,
            TrainerClass=recon_trainer_class,
            resume=None,
            log_dir=log_dir,
        )
        recon_test_metric = recon_pipeline.run_experiment(
            run_name=run_name, conf=conf, model_conf=model_conf_genval, seed=0
        )
        logger.info(f"Reconstructed test metric: {recon_test_metric};")
    if args.gen_val:
        generated_data_path = trainer.generate_data(train_supervised_loader)
        conf.train_path = generated_data_path
        print(conf.train_path)
        conf.valid_size = 0.1

        run_name = run_name
        total_epochs = args.gen_val_epoch
        model_conf_genval = model_configs_genval()
        log_dir = "./logs/generations/"
        trainer_class = deepcopy(get_trainer_class("rosbank"))
        # fix where the best model saves
        pipeline = SupervisedPipeline(
            run_name=run_name,
            device=args.device,
            total_epochs=total_epochs,
            conf=conf,
            model_conf=model_conf_genval,
            TrainerClass=trainer_class,
            resume=None,
            log_dir=log_dir,
        )
        generated_test_metric = pipeline.run_experiment(
            run_name=run_name, conf=conf, model_conf=model_conf_genval, seed=0
        )
        logger.info(f"Generated test metric: {generated_test_metric};")
