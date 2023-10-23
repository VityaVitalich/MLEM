import logging
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor as Pool

import torch
import sys
import os

sys.path.append("../")

from configs.data_configs.contrastive.rosbank import data_configs as rosbank_data
from configs.data_configs.contrastive.age import data_configs as age_data
from configs.data_configs.contrastive.physionet import data_configs as physionet_data
from configs.data_configs.contrastive.taobao import data_configs as taobao_data

from configs.model_configs.contrastive.age import model_configs as age_model
from configs.model_configs.contrastive.physionet import model_configs as physionet_model
from configs.model_configs.contrastive.rosbank import model_configs as rosbank_model
from configs.model_configs.contrastive.taobao import model_configs as taobao_model

from src.data_load.dataloader import create_data_loaders, create_test_loader
from src.trainers.trainer_contrastive import AccuracyTrainerContrastive, AucTrainerContrastive
from src.trainers.randomness import seed_everything
import src.models.base_models


def run_experiment(
        run_name, 
        device, 
        total_epochs, 
        conf, 
        model_conf, 
        TrainerClass, 
        resume, 
        log_dir, 
        seed=0, 
        console_log="warning",
        file_log="info",
    ):
    ### SETUP LOGGING ###
    ch = logging.StreamHandler()
    cons_lvl = getattr(logging, console_log.upper())
    ch.setLevel(cons_lvl)
    cfmt = logging.Formatter("{levelname:8} - {asctime} - {message}", style="{")
    ch.setFormatter(cfmt)

    (Path(log_dir) / run_name).mkdir(parents=True, exist_ok=True)
    run_name = f"{run_name}/seed_{seed}"
    log_file = Path(log_dir) / f"{run_name}.log"
    fh = logging.FileHandler(log_file)
    file_lvl = getattr(logging, file_log.upper())
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

    ### Fix randomness ###
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    model_conf.device = device
    conf.client_list_shuffle_seed = seed
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

    model = getattr(src.models.base_models, model_conf.model_name)
    net = model(model_conf=model_conf, data_conf=conf)
    opt = torch.optim.Adam(
        net.parameters(), model_conf.lr, weight_decay=model_conf.weight_decay
    )
    trainer = TrainerClass(
        model=net,
        optimizer=opt,
        train_loader=train_loader,
        val_loader=valid_loader,
        run_name=run_name,
        ckpt_dir=Path(log_dir).parent / "ckpt",
        ckpt_replace=True,
        ckpt_resume=resume,
        ckpt_track_metric="total_loss",
        metrics_on_train=False,
        total_epochs=total_epochs,
        device=device,
        model_conf=model_conf,
    )

    ### RUN TRAINING ###
    trainer.run()

    test_metrics = trainer.test(test_loader, train_supervised_loader)

    logger.removeHandler(fh)
    fh.close()
    return test_metrics

def run_experiment_helper(args):
    return run_experiment(*args)

def do_n_runs(run_name, device, total_epochs, conf, model_conf, TrainerClass, resume, log_dir, n_runs=3):
    run_name = f"{run_name}/{datetime.now():%F_%T}"

    result_list = []
    args = [
        (
            run_name,
            device,
            total_epochs,
            conf,
            model_conf,
            TrainerClass,
            resume,
            log_dir,
            seed,
            console_log,
            file_log,
        ) for seed in range(n_runs)
    ]
    with Pool(3) as p:
        result_list = p.map(run_experiment_helper, args)

    test_dict, train_dict, val_dict = {}, {}, {}
    for (test_metrics, train_metrics, val_metrics) in result_list:
        for data in zip(
            [test_dict, train_dict, val_dict], 
            [test_metrics, train_metrics, val_metrics]
        ):
            for k in data[1]:
                v = data[0].get(k, [])
                v += [round(data[1][k], 6)]
                data[0][k] = v

    summary_dict = {}
    for data in zip([test_dict, train_dict, val_dict], ["Test", "Train", "Val"]):
        for k, v in data[0].items():
            v += [np.mean(v), np.std(v)]
            summary_dict[f"({k}/{data[1]})"] = v
    summary_dict = dict(sorted(summary_dict.items()))
    summary_df = pd.DataFrame(summary_dict)
    summary_df.index = list(range(summary_df.shape[0] - 2)) + ["mean", "std"]
    summary_df.T.to_csv(Path(log_dir) / run_name / "results.csv")
    return summary_df

def do_grid(run_name, device, total_epochs, conf, model_conf, TrainerClass, resume, log_dir, n_runs=3):
    # grid_example = {
    #     "encoder": ["Identity", "TransformerEncoder"],
    #     "classifier_gru_hidden_dim": [16, 32, 64, 128],
    #     "encoder_norm": ["LayerNorm", "Identity"],
    #     "after_enc_dropout": [0, 0.3],
    # }
    assert not (Path(log_dir) / run_name).exists(), f"{Path(log_dir) / run_name} ALREADY EXISTS!!"
    res_dict = {}
    for GRU in [128, 64, 32, 16]:
        for TR in [True, False]:
            for DRP in [0, 0.3]:
                for BF in [False, True]:
                    model_conf["classifier_gru_hidden_dim"] = GRU
                    model_conf["encoder"] = ["Identity", "TransformerEncoder"][TR]
                    model_conf["after_enc_dropout"] = DRP
                    model_conf["encoder_norm"] = ["Identity", "LayerNorm"][TR]
                    model_conf["batch_first_encoder"] = BF
                    name = f"{run_name}/TR_{TR}_DRP_{DRP}_GRU_{GRU}_BF_{BF}"
                    res_dict[name] = do_n_runs(
                        name,
                        device,
                        total_epochs,
                        conf,
                        model_conf,
                        TrainerClass,
                        resume,
                        log_dir,
                        n_runs,
                    )

    return res_dict


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
    parser.add_argument("--device", help="torch device to run on", default="cuda")
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
        "--n-runs",
        help="number of runs with different seed",
        default=3,
        type=int,
    )
    parser.add_argument(
        "--dataset",
        help="dataset",
        type=str,
        default="physionet"
    )
    args = parser.parse_args()

    ### TRAINING SETUP ###
    run_name = args.run_name
    dataset = args.dataset
    conf = get_data_config(dataset)
    model_conf = get_model_config(dataset)
    TrainerClass = get_trainer_class(dataset)
    log_dir = Path(dataset) / args.log_dir

    # summary_df = do_n_runs(
    #     run_name,
    #     args.device,
    #     args.total_epochs,
    #     conf,
    #     model_conf,
    #     TrainerClass,
    #     args.resume,
    #     log_dir,
    #     args.n_runs,
    # )
    res = do_grid(
        run_name,
        args.device,
        args.total_epochs,
        conf,
        model_conf,
        TrainerClass,
        args.resume,
        log_dir,
        args.n_runs,
    )