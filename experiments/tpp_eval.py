from pipeline_contrastive import get_trainer_class as contrastive_trainer
from src.trainers.trainer_gen import GenTrainer
from src.trainers.trainer_sigmoid import SigmoidTrainer
import pandas as pd
from src.data_load.dataloader import create_data_loaders, create_test_loader
import src.models.gen_models
import src.models.base_models
import torch
from pathlib import Path

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
            lin_prob_test,
        ) = test_res
        metrics = {
            "train_metric": train_metric,
            "val_metric": supervised_val_metric,
            "test_metric": supervised_test_metric,
            "other_metric": fixed_test_metric,
            "lin_prob_val": lin_prob_test[0],
            "lin_prob_test": lin_prob_test[1],
            "lin_prob_fixed_test": lin_prob_test[2]
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
            "val_metric": val_metric,
            "test_metric": test_metric,
            "another_test_metric": another_test_metric,
            # "train_logist": train_logist,
            "lin_prob_val": val_logist,
            "lin_prob_test": test_logist,
            "lin_prob_fixed_test": another_test_logist,
            # "anisotropy": anisotropy,
            # "intrinsic_dimension": intrinsic_dimension,
        }
    else:
        raise NotImplementedError
    
def run_validation(setting, checkpoint_path, data_conf, model_conf):
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
    res = pd.DataFrame(index=prefixes)
    for prefix in prefixes:
        data_conf.train_path = init_train_path.with_name(prefix + init_train_path.name)
        data_conf.test_path = init_test_path.with_name(prefix + init_test_path.name)
        (
            train_supervised_loader,
            valid_supervised_loader,
            test_supervised_loader,
        ) = create_data_loaders(data_conf, pinch_test=True)
        fixed_test_loader = create_test_loader(data_conf)
        metrics = parse_test_res(trainer.test(
            train_supervised_loader,
            (valid_supervised_loader, test_supervised_loader, fixed_test_loader),
        ))
        for k in metrics:
            res.loc[prefix[:-1], k]
    return res

# def validate_seeds()