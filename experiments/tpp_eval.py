from pipeline_contrastive import get_trainer_class as contrastive_trainer
from src.trainers.trainer_gen import GenTrainer
from src.trainers.trainer_sigmoid import SigmoidTrainer
import pandas as pd
import numpy as np
from src.data_load.dataloader import create_data_loaders, create_test_loader
import src.models.gen_models
import src.models.base_models
import torch
from pathlib import Path
from experiments.utils import read_config
from experiments.utils import log_to_file
import copy
from argparse import ArgumentParser


def prepare_trainer(setting, checkpoint_path, data_conf, model_conf):
    if setting == "gen" or setting == "gen_contrastive":
        net = getattr(src.models.gen_models, model_conf.model_name)(
            model_conf=model_conf, data_conf=data_conf
        )
        # opt = torch.optim.Adam(
        #     net.parameters(), model_conf.lr, weight_decay=model_conf.weight_decay
        # )
        trainer = GenTrainer(
            model=net,
            optimizer=None,
            train_loader=None,
            val_loader=None,
            run_name="test",
            ckpt_dir=Path("tpp_test") / "ckpt",
            ckpt_replace=True,
            ckpt_resume=False,
            ckpt_track_metric="total_loss",
            metrics_on_train=False,
            total_epochs=0,
            device=model_conf.device,
            model_conf=model_conf,
            data_conf=data_conf,
        )
        trainer.load_ckpt(checkpoint_path)
        return trainer
    elif setting == "contrastive":
        net = getattr(src.models.base_models, model_conf.model_name)(
            model_conf=model_conf, data_conf=data_conf
        )
        TrainerClass = contrastive_trainer(data_conf)
        trainer = TrainerClass(
            model=net,
            optimizer=None,
            train_loader=None,
            val_loader=None,
            run_name="test",
            ckpt_dir=Path("tpp_test") / "ckpt",
            ckpt_replace=True,
            ckpt_resume=False,
            ckpt_track_metric="epoch",
            metrics_on_train=False,
            total_epochs=0,
            device=model_conf.device,
            model_conf=model_conf,
            data_conf=data_conf,
        )
        trainer.load_ckpt(checkpoint_path)
        return trainer
    elif setting == "sigmoid":
        net = getattr(src.models.gen_models, model_conf.model_name)(
            model_conf=model_conf, data_conf=data_conf
        )
        trainer = SigmoidTrainer(
            model=net,
            optimizer=None,
            train_loader=None,
            val_loader=None,
            run_name="test",
            ckpt_dir=Path("tpp_test") / "ckpt",
            ckpt_replace=True,
            ckpt_resume=False,
            ckpt_track_metric="total_loss",
            metrics_on_train=False,
            total_epochs=0,
            device=model_conf.device,
            model_conf=model_conf,
            data_conf=data_conf,
            contrastive_model=None, # TODO do for this
        )
        trainer.load_ckpt(checkpoint_path)
        return trainer
    else:
        raise NotImplementedError
    
def parse_test_res(setting, test_res):
    if setting == "gen" or setting == "gen_contrastive" or setting == "sigmoid":
        (
            train_metric,
            (supervised_val_metric, supervised_test_metric, fixed_test_metric),
            lin_prob_test, intrinsic_dimension, anisotropy
        ) = test_res
        metrics = {
            "train_metric": train_metric,
            "test_metric": fixed_test_metric,
            "lin_prob_fixed_test": lin_prob_test[2],
            "anisotropy": anisotropy,
            "intrinsic_dimension": intrinsic_dimension,
        }
        return metrics
    elif setting == "contrastive":
        (
            train_metric,
            (val_metric, test_metric, another_test_metric),
            train_logist,
            (val_logist, test_logist, another_test_logist),
            anisotropy,
            intrinsic_dimension,
        ) = test_res
        return {
            "train_metric": train_metric,
            "test_metric": another_test_metric,
            "lin_prob_fixed_test": another_test_logist,
            "anisotropy": anisotropy,
            "intrinsic_dimension": intrinsic_dimension,
        }
    else:
        raise NotImplementedError
    
def run_noise(setting, checkpoint_path, data_conf, model_conf):
    data_conf = copy.deepcopy(data_conf)
    data_conf.train.split_strategy = {"split_strategy": "NoSplit"}
    data_conf.val.split_strategy = {"split_strategy": "NoSplit"}
    data_conf.train.dropout = 0
    data_conf.valid_size = 0.0
    data_conf.test_size = 0.0
    # TODO set seed via checkpoint_path
    init_train_path = data_conf.train_path
    init_test_path = data_conf.test_path

    trainer = prepare_trainer(setting, checkpoint_path, data_conf, model_conf)
    prefixes = [
        "",
        "Drop_0.1_",
        "Drop_0.3_",
        "Drop_0.5_",
        "Drop_0.7_",
        "Permute_",
    ]
    res = pd.DataFrame({})
    for prefix in prefixes:
        data_conf.train_path = init_train_path.with_name(prefix + init_train_path.name)
        data_conf.test_path = init_test_path.with_name(prefix + init_test_path.name)
        (
            train_supervised_loader,
            valid_supervised_loader,
            test_supervised_loader,
        ) = create_data_loaders(data_conf, pinch_test=True)
        fixed_test_loader = create_test_loader(data_conf)
        metrics = parse_test_res(setting, trainer.test(
            train_supervised_loader,
            (valid_supervised_loader, test_supervised_loader, fixed_test_loader),
        ))
        prefix = "ORIG" if prefix == "" else prefix
        print(f"{prefix} DONE")
        print(metrics)
        for k in metrics:
            res.loc[k, prefix[:-1]] = metrics[k]
    return res

def run_tpp(setting, checkpoint_path, data_conf, model_conf):
    data_conf = copy.deepcopy(data_conf)
    data_conf.train.split_strategy = {"split_strategy": "NoSplit"}
    data_conf.val.split_strategy = {"split_strategy": "NoSplit"}
    data_conf.train.dropout = 0
    data_conf.valid_size = 0.0
    data_conf.test_size = 0.0
    # TODO set seed via checkpoint_path
    init_train_path = data_conf.train_path
    init_test_path = data_conf.test_path
    data_conf.train_path = init_train_path.with_name("tpp_" + init_train_path.name)
    data_conf.test_path = init_test_path.with_name("tpp_" + init_test_path.name)

    if ('physionet' in str(init_train_path)) or ('pendulum' in str(init_train_path)):
        targets = ["time"]
    else:
        targets = ["event", "time"]

    res = pd.DataFrame({})
    if "age" in str(init_train_path):
        Ns = [1, 2]
    elif "alpha" in str(init_train_path):
        Ns = [1, 2]
    elif "rosbank" in str(init_train_path):
        Ns = [1, 2]
    elif "physionet" in str(init_train_path):
        Ns = [1, 2]
    elif "pendulum" in str(init_train_path):
        Ns = [1, 2]
    elif "taobao" in str(init_train_path):
        Ns = [1, 2]
    elif "amex" in str(init_train_path):
        Ns = [1, 2]
    else:
        raise NotImplementedError
    for n in Ns:
        for target in targets:
            data_conf.track_metric = "accuracy" if target == "event" else "mse"
            data_conf.features.target_col = f"{n}_{target}"
            trainer = prepare_trainer(setting, checkpoint_path, data_conf, model_conf)
            (
                train_supervised_loader,
                valid_supervised_loader,
                test_supervised_loader,
            ) = create_data_loaders(data_conf, pinch_test=True)
            fixed_test_loader = create_test_loader(data_conf)
            metrics = parse_test_res(setting, trainer.test(
                train_supervised_loader,
                (valid_supervised_loader, test_supervised_loader, fixed_test_loader),
            ))
            print(f"{n}_{target} DONE")
            print(metrics)
            for k in metrics:
                res.loc[k, f"{n}_{target}"] = metrics[k]
    return res

def run_all(DATA_C, MODEL_C, device, setting, checkpoint_path, noise_only=False):
    print("START", DATA_C, MODEL_C, device, setting, checkpoint_path)
    data_conf = read_config(DATA_C, "data_configs")
    model_conf = read_config(MODEL_C, "model_configs")
    model_conf.device = device
    res = []
    funcs = [run_noise] if noise_only else [run_tpp, run_noise]
    for func in funcs: 
        with log_to_file("/dev/null", file_lvl="info", cons_lvl="info"):
            res += [func(
                setting, 
                checkpoint_path,
                data_conf,
                model_conf,
            )]
    res = pd.concat(res, axis=1)
    res.to_csv(Path(checkpoint_path).parent / "tpp_noise.csv")
    return res

def parse_seeds(pathes):
    res = [pd.read_csv(path, index_col=0) for path in pathes]
    new_df = pd.DataFrame(columns=res[0].index)
    for col in res[0].columns:
        for metric in res[0].index:
            metric_seeds = []
            for df in res:
                metric_seeds += [df.loc[metric, col]]
            new_df.loc[col, metric] = f"{np.mean(metric_seeds).round(4)},{np.std(metric_seeds).round(3)}"
    print(new_df)
    return new_df
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--configuration", type=str)
    parser.add_argument('--ckpt_0', type=str)
    parser.add_argument('--ckpt_1', type=str)
    parser.add_argument('--ckpt_2', type=str)
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--data_config', type=str)
    parser.add_argument('--noise_only', default=False)
    args = parser.parse_args()
    configurations = []
    configurations += [
        [args.data_config,
        args.model_config,
        args.device,
        args.configuration,
        eval(f'args.ckpt_{i}'),
        ] for i in range(3)
    ]
    res = []
    for i in range(3):
        res += [run_all(*configurations[i], noise_only=args.noise_only)]
    print(res)
    paths = [Path(eval(f'args.ckpt_{i}')).parent / 'tpp_noise.csv' for i in range(3)]
    df = parse_seeds(paths)

    save_path = Path(eval(f'args.ckpt_{0}')).parent / 'result.csv'
    df.to_csv(save_path)
