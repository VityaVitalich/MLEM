import copy
import logging
from concurrent.futures import ProcessPoolExecutor as Pool
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.samplers import TPESampler
from src.trainers.randomness import seed_everything


class Pipeline:
    def __init__(
        self,
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
        self.run_name = run_name
        self.total_epochs = total_epochs
        self.conf = conf
        self.model_conf = model_conf
        self.model_conf.device = device
        self.TrainerClass = TrainerClass
        self.resume = resume
        self.log_dir = log_dir
        self.console_log = console_log
        self.file_log = file_log
        """
        TrainerClass - class from src.trainers
        """

    def setup_logging(
        self,
        run_name,
        log_dir,
        console_log="warning",
        file_log="info",
    ):
        ### SETUP LOGGING ###
        ch = logging.StreamHandler()
        cons_lvl = getattr(logging, console_log.upper())
        ch.setLevel(cons_lvl)
        cfmt = logging.Formatter("{levelname:8} - {asctime} - {message}", style="{")
        ch.setFormatter(cfmt)

        (Path(log_dir) / run_name).parent.mkdir(parents=True, exist_ok=True)
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
        return logger, ch, fh

    def run_experiment(self, run_name=None, conf=None, model_conf=None, seed=0):
        run_name = f"{run_name or self.run_name}/seed_{seed}"
        conf = conf or copy.deepcopy(self.conf)
        model_conf = model_conf or copy.deepcopy(self.model_conf)

        conf.client_list_shuffle_seed = seed
        logger, ch, fh = self.setup_logging(
            run_name, self.log_dir, self.console_log, self.file_log
        )
        ### Fix randomness ###
        seed_everything(
            conf.client_list_shuffle_seed,
            avoid_benchmark_noise=True,
            only_deterministic_algorithms=False,
        )
        metrics = self._train_eval(run_name, conf, model_conf)

        logger.removeHandler(fh)
        logger.removeHandler(ch)
        fh.close()
        return metrics

    def _run_experiment_helper(self, *args, **kwargs):
        return self.run_experiment(*args, **kwargs)

    def do_n_runs(
        self,
        run_name=None,
        conf=None,
        model_conf=None,
        n_runs=3,
    ):
        """
        Expects run_experiment() to return dict like {metric_name: metric_value}
        """
        run_name = f"{run_name or self.run_name}/{datetime.now():%F_%T}"
        conf = conf or copy.deepcopy(self.conf)
        model_conf = model_conf or copy.deepcopy(self.model_conf)

        args = [(run_name, conf, model_conf, seed) for seed in range(n_runs)]
        if len(args) > 1:
            with Pool(3) as p:
                result_list = p.map(self._run_experiment_helper, args)
        else:
            result_list = [self._run_experiment_helper(args[0])]

        summary_dict = {k: [] for k in result_list[0]}
        for metrics in result_list:
            for metric_name in metrics:
                summary_dict[metric_name].append(metrics[metric_name])

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
        n_trials == n_trials total by this function call(doesn't affect parallel runs).
        n_runs - better not torch it
        """
        sampler = TPESampler(
            # seed=0, important to NOT specify, otherwise parallel scripts repeat themself
            multivariate=True,
            group=True,  # Very usefull, allows to use conditional subsets of parameters.
            n_startup_trials=n_startup_trials,
        )
        try:
            first = False
            study = optuna.load_study(
                study_name=f"{self.log_dir / self.run_name}",
                storage="sqlite:///example.db",
            )
        except:
            first = True
        study = optuna.create_study(
            storage="sqlite:///example.db",
            sampler=sampler,
            study_name=f"{self.log_dir / self.run_name}",
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
                    "loss.loss_fn": "ContrastiveLoss",
                    "encoder_norm": "LayerNorm",
                }
            )  # TODO move somewhere else
        for request in request_list:
            study.enqueue_trial(request)

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
        conf = copy.deepcopy(self.conf)
        model_conf = copy.deepcopy(self.model_conf)
        trial, model_conf, conf = self._param_grid(trial, model_conf, conf)

        print(trial.number, trial.params)
        name = f"{self.run_name}/{trial.number}"
        print("RUN NAME:", name)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        trial.set_user_attr(
            "memory_before", torch.cuda.max_memory_reserved(self.device)
        )
        summary_df = self.do_n_runs(
            run_name=name,
            conf=conf,
            model_conf=model_conf,
            n_runs=n_runs,
        )
        trial.set_user_attr("memory_after", torch.cuda.max_memory_reserved(self.device))

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
