import copy
import logging
from concurrent.futures import ProcessPoolExecutor as Pool
from multiprocessing import set_start_method

# try:
#     set_start_method("spawn")
# except:
#     pass
import os
import sys
import subprocess
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict
import time

import numpy as np
import pandas as pd
import torch
from experiments.utils import log_to_file
from src.trainers.randomness import seed_everything


class Pipeline:
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
        grid_name="",
        console_lvl="warning",
        file_lvl="info",
        resume_list=None
    ):
        """
        TrainerClass - class from src.trainers
        """

        self.run_name = run_name
        self.total_epochs = total_epochs
        self.data_conf = copy.deepcopy(data_conf)
        self.model_conf = copy.deepcopy(model_conf)
        self.model_conf.device = device
        self.device = device
        self.TrainerClass = TrainerClass
        self.resume = resume
        self.log_dir = Path(log_dir)
        self.grid_name = grid_name
        self.console_lvl = console_lvl
        self.file_lvl = file_lvl
        self.resume_list = resume_list
        """
        TrainerClass - class from src.trainers
        """

    def run_experiment(
        self, run_name=None, conf=None, model_conf=None, seed=0
    ) -> Dict[str, float]:
        """
        metrics = {"metric_name": metric_value(float)}.
        """
        run_name = f"{run_name or self.run_name}/seed_{seed}"
        conf = conf or copy.deepcopy(self.data_conf)
        model_conf = model_conf or copy.deepcopy(self.model_conf)

        conf.client_list_shuffle_seed += seed

        log_file = self.log_dir / run_name / "log"
        log_file.parent.mkdir(exist_ok=True, parents=True)

        with log_to_file(log_file, self.file_lvl, self.console_lvl):
            ### Fix randomness ###
            seed_everything(
                conf.client_list_shuffle_seed,
                avoid_benchmark_noise=True,
                only_deterministic_algorithms=False,
            )
            metrics = self._train_eval(run_name, conf, model_conf)
        return metrics

    def _run_experiment_helper(self, args):
        torch.cuda.empty_cache()
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(self.device)
        # print(f"Sleeping for {args[-1]} minutes")
        # time.sleep(60 * args[-1])
        metrics = self.run_experiment(*args)
        metrics["memory_after"] = torch.cuda.max_memory_reserved(self.device) / (
            2**20
        )
        return metrics

    def do_n_runs(
        self,
        run_name=None,
        conf=None,
        model_conf=None,
        n_runs=3,
        max_workers=3,
    ):
        """
        Expects run_experiment() to return dict like {metric_name: metric_value}
        """
        run_name = f"{run_name or self.run_name}"  # /{datetime.now():%F_%T}"
        conf = conf or copy.deepcopy(self.data_conf)
        model_conf = model_conf or copy.deepcopy(self.model_conf)

        args = [(run_name, conf, model_conf, seed) for seed in range(n_runs)]
        if len(args) > 1:
            with Pool(max_workers) as p:
                result_list = list(p.map(self._run_experiment_helper, args))
        else:
            result_list = [self._run_experiment_helper(args[0])]

        summary_dict = {}
        for metrics in result_list:
            for metric_name in metrics:
                v = summary_dict.get(metric_name, []) + [metrics[metric_name]]
                summary_dict[metric_name] = v

        for k in summary_dict:
            summary_dict[k] = summary_dict[k] + [
                np.mean(summary_dict[k]),
                np.std(summary_dict[k]),
            ]

        summary_dict = dict(sorted(summary_dict.items()))
        summary_df = pd.DataFrame(summary_dict)
        summary_df.index = list(range(summary_df.shape[0] - 2)) + ["mean", "std"]
        summary_df.T.to_csv(Path(self.log_dir) / run_name / "results.csv")
        return summary_df

    def optuna_setup(
        self,
        target_metric="val_metric",
        request_list=[],
        n_startup_trials=3,
        n_trials=50,
        n_runs=3,
    ):
        """
        Set target_metric according to _train_eval().
        request_list - list of dicts where {key:value} is {trial_parameter_name:parameter_value}
        n_startup_trials == n_random_trials
        n_trials == n_trials to make in total by this function call(doesn't affect parallel runs).
        n_runs - better not torch it
        """

        import optuna
        from optuna.samplers import TPESampler
        from optuna.storages import JournalFileStorage, JournalStorage

        optuna.logging.get_logger("optuna").addHandler(
            logging.StreamHandler(sys.stdout)
        )
        optuna.logging.enable_propagation()
        sampler = TPESampler(
            # seed=0, important to NOT specify, otherwise parallel scripts repeat themself
            multivariate=True,
            group=True,  # Very usefull, allows to use conditional subsets of parameters.
            n_startup_trials=n_startup_trials,
        )
        (self.log_dir / self.run_name).mkdir(exist_ok=True, parents=True)

        storage = JournalStorage(
            JournalFileStorage(f"{self.log_dir / self.run_name}/study.log")
        )
        study = optuna.create_study(
            storage=storage,
            sampler=sampler,
            study_name=f"{self.log_dir / self.run_name}",
            direction="maximize",
            load_if_exists=True,
        )

        for request in request_list:
            study.enqueue_trial(request, skip_if_exists=True)

        study.optimize(
            lambda trial: self._objective(trial, n_runs, target_metric),
            n_trials=n_trials,
        )
        return study

    def _objective(
        self,
        trial,
        n_runs=3,
        target_metric="val_metric",
    ):
        conf = copy.deepcopy(self.data_conf)
        model_conf = copy.deepcopy(self.model_conf)
        trial, model_conf, conf = self._param_grid(trial, model_conf, conf)

        print(trial.number, trial.params)
        name = f"{self.run_name}/{trial.number}"
        print("RUN NAME:", name)

        try:
            summary_df = self.do_n_runs(
                run_name=name,
                conf=conf,
                model_conf=model_conf,
                n_runs=n_runs,
            )
        except Exception as e:
            (Path(self.log_dir) / name).mkdir(exist_ok=True, parents=True)
            (Path(self.log_dir) / name / "ERROR.txt").write_text(traceback.format_exc())
            print(traceback.format_exc())
            raise e
        (Path(self.log_dir) / name / "params.txt").write_text(
            str(trial.params).replace(", ", ",\n")
        )

        for k in summary_df:
            trial.set_user_attr(f"{k}_mean", summary_df.loc["mean", k])
            trial.set_user_attr(f"{k}_std", summary_df.loc["std", k])

        return (
            summary_df.loc["mean", target_metric] - summary_df.loc["std", target_metric]
        )

    def _train_eval(self, run_name, conf, model_conf):
        """
        Returns metrics dict like {"train_acc": metric_value, "val_acc": metric_value}
        Make sure that metric_value is going to be MAXIMIZED (higher -> better)
        """
        raise NotImplementedError

    def _param_grid(self, trial, model_conf, conf):
        """
        Returns trial, model_conf, conf with changed values.
        """
        raise NotImplementedError
