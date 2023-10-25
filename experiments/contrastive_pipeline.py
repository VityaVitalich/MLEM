import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor as Pool
from utils import parse_args

import torch
import sys

import optuna
from optuna.samplers import TPESampler
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

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
    """
    TrainerClass - class from src.trainers
    """
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
        ckpt_track_metric="epoch",
        metrics_on_train=False,
        total_epochs=total_epochs,
        device=device,
        model_conf=model_conf,
    )

    ### RUN TRAINING ###
    trainer.run()

    metrics = trainer.test(test_loader, train_supervised_loader)

    logger.removeHandler(fh)
    logger.removeHandler(ch)
    fh.close()
    return metrics

def run_experiment_helper(args):
    return run_experiment(*args)

def do_n_runs(
    run_name, 
    device, 
    total_epochs, 
    conf, 
    model_conf, 
    TrainerClass, 
    resume, 
    log_dir, 
    n_runs=3,
    console_log="warning",
    file_log="info",
):
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
    if len(args) > 1:
        with Pool(3) as p:
            result_list = p.map(run_experiment_helper, args)
    else:
        result_list = [run_experiment_helper(args[0])]

    summary_dict = dict()
    for d in result_list:
        for k in d:
            l = summary_dict.get(k, [])
            l += [np.mean(d[k])]
            summary_dict[k] = l
    for k in summary_dict:
        summary_dict[k] = summary_dict[k] + [np.mean(summary_dict[k]), np.std(summary_dict[k])]

    summary_dict = dict(sorted(summary_dict.items()))
    summary_df = pd.DataFrame(summary_dict)
    summary_df.index = list(range(summary_df.shape[0] - 2)) + ["mean", "std"]
    summary_df.T.to_csv(Path(log_dir) / run_name / "results.csv")
    return summary_df

def objective(
    trial,
    run_name,
    device,
    total_epochs,
    conf,
    model_conf,
    TrainerClass,
    resume,
    log_dir,
    console_log="warning",
    file_log="info",
):
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
        model_conf["num_heads_enc"] = trial.suggest_categorical("num_heads_enc", [1, 2, 4])
    else:
        model_conf["num_heads_enc"] = 1
    
    if model_conf["encoder"] == "TransformerEncoder":
        model_conf["encoder_norm"] = trial.suggest_categorical("encoder_norm", ["Identity", "LayerNorm"]) # important to know corr to encoder
    elif model_conf["encoder"] == "Identity":
        model_conf["encoder_norm"] = "Identity"

    for param in (
        ("loss.loss_fn", ["ContrastiveLoss"]), #, "InfoNCELoss", "DecoupledInfoNCELoss", "RINCELoss", "DecoupledPairwiseInfoNCELoss"]),
        ("loss.projector", ["Identity", "Linear", "MLP"]),
        ("loss.project_dim", [32, 64, 128, 256]),
    ):
        model_conf.loss[param[0].split(".")[1]] = trial.suggest_categorical(param[0], param[1])
    if model_conf.loss.loss_fn == "ContrastiveLoss":
        model_conf.loss.margin = trial.suggest_categorical("loss.margin", [0.0, 0.1, 0.3, 0.5, 1.0])
    else:
        model_conf.loss.temperature = trial.suggest_categorical("loss.temperature", [0.01, 0.03, 0.1, 0.3, 1.0])
    if model_conf.loss.loss_fn == "InfoNCELoss":
        model_conf.loss.angular_margin = trial.suggest_categorical("loss.angular_margin", [0.0, 0.3, 0.5, 0.7])
    elif model_conf.loss.loss_fn == "RINCELoss":
        model_conf.loss.q = trial.suggest_categorical("loss.q", [0.01, 0.03, 0.1, 0.3])
        model_conf.loss.lam = trial.suggest_categorical("loss.lam", [0.003, 0.01, 0.03, 0.1, 0.3] )
    print(trial.number, trial.params)
    name = f"{run_name}/{trial.number}"
    print("RUN NAME:", name)
    res_dict = run_experiment(
        name,
        device,
        total_epochs,
        conf,
        model_conf,
        TrainerClass,
        resume,
        log_dir,
        console_log=console_log,
        file_log=file_log,
    )
    trial.set_user_attr("train_metric", np.mean(res_dict[[k for k in res_dict if "train" in k][0]]))

    keys = [k for k in res_dict if "test" in k]
    assert len(keys) == 1
    return np.mean(res_dict[keys[0]])

def optuna_setup(*args, **kwargs):
    sampler = TPESampler(
        # seed=0, important to NOT specify, otherwise parallel scripts repeat themself
        multivariate=True,
        group=True, # Very usefull, allows to use conditional subsets of parameters.
        n_startup_trials=3,
    )

    try:
        first = False
        study = optuna.load_study(study_name=f"{log_dir / run_name}", storage="sqlite:///example.db")
    except:
        first = True
    study = optuna.create_study(
        storage="sqlite:///example.db",
        sampler=sampler,
        study_name=f"{log_dir / run_name}",
        direction="maximize",
        load_if_exists=True,
    )

    if first:
        print("First!!!!!")
        study.enqueue_trial(
            {
                "features_emb_dim": 32,
                "classifier_gru_hidden_dim": 128,
                "encoder": "TransformerEncoder",
                "num_enc_layers": 2,
                "use_numeric_emb": True,
                "loss.projector": "Linear",
                "loss.project_dim": 256,     
                "num_heads_enc": 4,       
                'loss.loss_fn': 'ContrastiveLoss',
                "encoder_norm": "LayerNorm",
            }
        )

    study.optimize(
        lambda trial: objective(trial, *args, **kwargs), 
        n_trials=50, 
    )

    trial = study.best_trial
    print("Number of finished trials: ", len(study.trials), "Best trial, Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

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
    run_name = args.run_name
    dataset = args.dataset
    conf = get_data_config(dataset)
    model_conf = get_model_config(dataset)
    TrainerClass = get_trainer_class(dataset)
    log_dir = Path(dataset) / args.log_dir

    summary_df = optuna_setup(
        run_name=run_name,
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
    print(summary_df)
    # res = do_grid(
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
