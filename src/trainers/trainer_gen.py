import logging
from typing import Any, Dict, List, Literal, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import os
import pandas as pd

from ..models.mTAND.model import MegaNetCE
from ..data_load.dataloader import PaddedBatch
from .base_trainer import BaseTrainer, _CyclicalLoader
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from ..models.model_utils import (
    out_to_padded_batch,
    calc_anisotropy,
    calc_intrinsic_dimension,
)
from sklearn.linear_model import LinearRegression, LogisticRegression


params_fast = {
    "n_estimators": 200,
    "boosting_type": "gbdt",
    "subsample": 0.5,
    "subsample_freq": 1,
    "learning_rate": 0.02,
    "feature_fraction": 0.75,
    "max_depth": 6,
    "lambda_l1": 1,
    "lambda_l2": 1,
    "min_data_in_leaf": 50,
    "random_state": 42,
    "n_jobs": 16,
    "reg_alpha": None,
    "reg_lambda": None,
    "colsample_bytree": None,
    "min_child_samples": None,
    "verbosity": -1,
}

params_strong = {
    "n_estimators": 1000,
    "boosting_type": "gbdt",
    # "objective": "binary",
    # "metric": "auc",
    "subsample": 0.75,
    "subsample_freq": 1,
    "learning_rate": 0.02,
    "feature_fraction": 0.75,
    "max_depth": 12,
    "lambda_l1": 1,
    "lambda_l2": 1,
    "min_data_in_leaf": 50,
    "random_state": 42,
    "n_jobs": 16,
    "num_leaves": 50,
    "reg_alpha": None,
    "reg_lambda": None,
    "colsample_bytree": None,
    "min_child_samples": None,
    "verbosity": -1,
}

logger = logging.getLogger("event_seq")


class GenTrainer(BaseTrainer):
    def compute_metrics(
        self,
        model_outputs: List[Any],
        ground_truths: List[Any],  # pyright: ignore unused
    ) -> Dict[str, Any]:
        """Compute metrics based on model output.

        The function is used to compute model metrics. For further logging and
        and checkpoint tracking. Any metrics could be logged, but only scalar metrics
        are used to track checkpoints.

        Args:
            model_outputs: as is stacked model outputs during train or validation stage.
            ground_truths: as is stacked collected labels during train or validation
                stage.

        Returns:
            A dict of metric name and metric value(s).
        """
        # assert isinstance(self.model, MegaNetCE)
        # loss_dicts = [
        #     self.model.loss(it, gt) for it, gt in zip(model_outputs, ground_truths)
        # ]
        # losses_dict = {
        #     k: np.mean([d[k].item() for d in loss_dicts]) for k in loss_dicts[0]
        # }

        # return losses_dict
        raise NotImplementedError

    def compute_loss(
        self,
        model_output: Any,
        ground_truth: Tuple[int, int],  # pyright: ignore unused
    ) -> torch.Tensor:
        """Compute loss for backward.

        The function is called every iteration in training loop to compute loss.

        Args:
            model_output: raw model output as is.
            ground_truth: tuple of raw idx and raw ground truth label from dataloader.
        """
        # assert isinstance(self.model, MegaNet)
        losses = self.model.loss(model_output, ground_truth)

        for k, v in losses.items():
            losses[k] = torch.nan_to_num(v, nan=2000, posinf=1e8, neginf=-1e8)
        return losses["total_loss"]

    def log_metrics(
        self,
        phase: Literal["train", "val"],
        metrics: Union[Dict[str, Any], None] = None,
        epoch: Union[int, None] = None,
        losses: Union[List[float], None] = None,  # pyright: ignore unused
        iterations: Union[List[int], None] = None,  # pyright: ignore unused
    ):
        """Log metrics.

        The metrics are computed based on the whole epoch data, so the granularity of
        metrics is epoch, so when the metrics are not None, the epoch is not None to.
        The loss is computed every iteraton, so when the loss values are passed, the
        corresponding iterations are also passed to the function. The metrics are
        computed on validation phase, but can also be computed for train phase. The
        loss is computed only during train phase to report the validation loss, compute
        it in the `compute_metrics` function.

        Args:
            phase: wether the metrics were collected during train or validatoin.
            metrics: a dict that is returned from `compute_metrics` every epoch.
            epoch: number of epoch after which the metrics were computed.
            losses: a list of loss values.
            iterations: a list of iteration number for corresponding loss values.
        """
        if metrics is not None:
            logger.info(f"Epoch: {epoch}; metrics on {phase}: {metrics}")

    def test(
        self, train_supervised_loader: DataLoader, other_loaders: List[DataLoader]
    ) -> Dict[str, float]:
        """
        Logs test metrics with self.compute_metrics
        """
        logger.info("Test started")
        predict_limit = self._data_conf.get("predict_limit", None)
        train_out, train_gts = self.get_embeddings(train_supervised_loader, predict_limit)
        other_outs, other_gts = [], []
        for other_loader in other_loaders:
            other_out, other_gt = (
                self.get_embeddings(other_loader, predict_limit) if len(other_loader) > 0 else (None, None)
            )
            other_outs.append(other_out), other_gts.append(other_gt)

        train_embeddings = train_out
        other_embeddings = [
            other_out if other_out is not None else None
            for other_out in other_outs
        ]
        anisotropy = calc_anisotropy(train_embeddings, other_embeddings).item()
        logger.info("Anisotropy: %s", str(anisotropy))

        intrinsic_dimension = calc_intrinsic_dimension(
            train_embeddings, other_embeddings
        )
        logger.info("Intrinsic Dimension: %s", str(intrinsic_dimension))

        train_metric, other_metrics, lin_prob_metrics = self.compute_test_metric(
            train_embeddings, train_gts, other_embeddings, other_gts
        )
        logger.info("Train metrics: %s", str(train_metric))
        logger.info(
            "Validation, supervised Test, Fixed Test Metrics: %s", str(other_metrics)
        )
        logger.info(
            "LinProb Validation, supervised Test, Fixed Test Metrics: %s",
            str(lin_prob_metrics),
        )

        logger.info("Test finished")
        return train_metric, other_metrics, lin_prob_metrics, intrinsic_dimension, anisotropy

    def compute_test_metric(
        self,
        train_embeddings,
        train_gts,
        other_embeddings,
        other_gts,
    ):
        train_labels = torch.cat([gt[1].cpu() for gt in train_gts]).numpy()
        train_embeddings = torch.cat(train_embeddings).cpu().numpy()

        other_labels, other_embeddings_new = [], []
        for other_gt in other_gts:
            other_labels.append(
                torch.cat([gt[1].cpu() for gt in other_gt]).numpy()
                if other_gt is not None
                else None
            )
        for other_embedding in other_embeddings:
            other_embeddings_new.append(
                torch.cat(other_embedding).cpu().numpy()
                if other_embedding is not None
                else None
            )

        if hasattr(self._data_conf, "main_metric"):
            metric = self._data_conf.main_metric
        elif hasattr(self._data_conf, "track_metric"):
            metric = self._data_conf.track_metric
        else:
            raise NotImplementedError("no metric name provided")
        
        if np.unique(train_labels).shape[0] < 15:
            params = params_strong.copy()
            logist_params = {"solver": "saga"}
        else:
            params = params_fast.copy()
            logist_params = {"n_jobs":-1, "multi_class": "ovr", "solver": "saga"}
        if metric == "roc_auc":
            params["objective"] = "binary"
            params["metric"] = "auc"
            model = LGBMClassifier(
                **params,
            )
            lin_prob = LogisticRegression(**logist_params)
        elif metric == "accuracy":
            model = LGBMClassifier(
                **params,
            )
            lin_prob = LogisticRegression(**logist_params)
        elif metric == "mse":
            params["objective"] = "regression"
            model = LGBMRegressor()
            lin_prob = LinearRegression()
        else:
            raise NotImplementedError(f"Unknown objective {metric}")

        def get_metric(model, x, target):
            if metric == "roc_auc":
                return roc_auc_score(target, model.predict_proba(x)[:, 1])
            elif metric == "accuracy":
                return accuracy_score(target, model.predict(x))
            elif metric == "mse":
                return mean_squared_error(target, model.predict(x))
            else:
                raise NotImplementedError(f"Unknown objective {metric}")

        preprocessor = MaxAbsScaler()
        train_embeddings_tr = preprocessor.fit_transform(train_embeddings)
        model.fit(train_embeddings_tr, train_labels)
        lin_prob.fit(train_embeddings_tr, train_labels)

        train_metric = get_metric(model, train_embeddings_tr, train_labels)
        other_metrics = []
        lin_prob_metrics = []
        for i, (other_embedding, other_label) in enumerate(
            zip(other_embeddings_new, other_labels)
        ):
            if other_embedding is not None:
                other_embedding_proccesed = preprocessor.transform(other_embedding)
                other_metrics.append(
                    get_metric(model, other_embedding_proccesed, other_label)
                )
                lin_prob_metrics.append(
                    get_metric(lin_prob, other_embedding_proccesed, other_label)
                )
            else:
                other_metrics.append(0)
                lin_prob_metrics.append(0)

        return train_metric, other_metrics, lin_prob_metrics
    
    def get_embeddings(self, loader, limit=None):
        counter = 0
        self._model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for inp, gt in tqdm(loader):
                gts.append(gt.to(self._device))
                inp = inp.to(self._device)
                out = self._model(inp)
                out = self.dict_to_cpu(out)
                preds.append(out["latent"])
                counter += loader.batch_size
                if limit and counter > limit:
                    break
        return preds, gts

    def predict(
        self, loader: DataLoader, limit: int = None
    ) -> Tuple[List[Any], List[Any]]:
        counter = 0
        self._model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for inp, gt in tqdm(loader):
                gts.append(gt.to(self._device))
                inp = inp.to(self._device)
                out = self._model(inp)
                out = self.dict_to_cpu(out)
                out["gt"].pop("input_batch")
                out.pop("all_latents", None)
                preds.append(out)
                counter += loader.batch_size
                if limit and counter > limit:
                    break
        return preds, gts

    def validate(self) -> None:
        assert self._val_loader is not None, "Set a val loader first"

        logger.info("Epoch %04d: validation started", self._last_epoch + 1)
        self._model.eval()
        loss_dicts = []
        with torch.no_grad():
            for inp, gt in tqdm(self._val_loader):
                inp = inp.to(self._device)
                model_output = self._model(inp)
                loss_dicts.append(self._model.loss(model_output, gt))

        self._metric_values = {
            k: np.mean([d[k].item() for d in loss_dicts]) for k in loss_dicts[0]
        }
        logger.info(
            "Epoch %04d: validation metrics: %s",
            self._last_epoch + 1,
            str(self._metric_values),
        )
        logger.info("Epoch %04d: validation finished", self._last_epoch + 1)

    @staticmethod
    def dict_to_cpu(d):
        out = {}
        for k, val in d.items():
            if isinstance(val, dict):
                out[k] = GenTrainer.dict_to_cpu(val)
            else:
                out[k] = val.to("cpu")

        return out

    def reconstruct_data(self, train_supervised_loader):
        limit = self._data_conf.get("recon_limit", None)

        logger.info("Reconstruction started")
        train_out, train_gts = self.predict(train_supervised_loader, limit=limit)
        logger.info("Reconstructions convertation started")

        reconstructed_data = self.output_to_df(train_out, train_gts)
        logger.info("Reconstructions converted")
        save_path = (
            Path(self._ckpt_dir) / f"{self._run_name}" / "reconstructed_data.parquet"
        )

        reconstructed_data.to_parquet(save_path)
        logger.info("Reconstructions saved")
        return save_path

    def generate(
            self, loader: DataLoader, limit: int = None
    ) -> Tuple[List[Any], List[Any]]:
        counter = 0
        self._model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for inp, gt in tqdm(loader):
                gts.append(gt.to(self._device))
                inp = inp.to(self._device)
                out = self._model.generate(inp, self._model_conf.gen_len)
                out = self.dict_to_cpu(out)
                preds.append(out)

                counter += loader.batch_size
                if limit and counter > limit:
                    break
        return preds, gts

    def generate_data(self, train_supervised_loader):
        limit = self._data_conf.get("gen_limit", None)
        logger.info(f"Generation started with limit {limit}")
        train_out, train_gts = self.generate(train_supervised_loader, limit=limit)
        logger.info("Predictions convertation started")

        generated_data = self.output_to_df(
            train_out, train_gts, use_generated_time=True
        )
        logger.info("Predictions converted")
        save_path = (
            Path(self._ckpt_dir) / f"{self._run_name}" / "generated_data.parquet"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)

        generated_data.to_parquet(save_path)
        logger.info("Predictions saved")
        return save_path

    def output_to_df(self, outs, gts, use_generated_time=False):
        df_dic = {
            "event_time": [],
            "trx_count": [],
            self._data_conf.features.target_col: [],
        }
        for feature in self._data_conf.features.embeddings.keys():
            df_dic[feature] = []

        for feature in self._data_conf.features.numeric_values.keys():
            df_dic[feature] = []
        shift = int(self._data_conf.get("shift_embedding", True))
        for out, gt in zip(outs, gts):
            for key, val in out["pred"].items():
                if key in self._data_conf.features.embeddings.keys():
                    df_dic[key].extend((val.cpu().argmax(dim=-1) - shift).tolist())
                elif key in self._data_conf.features.numeric_values.keys():
                    df_dic[key].extend(val.cpu().squeeze(-1).tolist())

            if self._model_conf.use_log_delta and (
                not use_generated_time
            ):  # use generated time equals to gen. not use to recon
                pred_delta = torch.exp(out["pred"]["delta"]).cumsum(1)
            else:
                pred_delta = out["pred"]["delta"].cumsum(1)

            df_dic["event_time"].extend(pred_delta.tolist())
            if use_generated_time:
                df_dic["trx_count"].extend((pred_delta != -1).sum(dim=1).tolist())
            else:
                df_dic["trx_count"].extend(
                    (out["gt"]["time_steps"] != -1).sum(dim=1).tolist()
                )
            # num_numeric = len(self._data_conf.features.numeric_values.keys())
            # numeric_pred = pred[:, :, -num_numeric:]
            # for i in range(num_numeric):
            #     cur_key = self.order[i]
            #     cur_val = numeric_pred[:, :, i].cpu().tolist()
            #     df_dic[cur_key].extend(cur_val)

            df_dic[self._data_conf.features.target_col].extend(gt[1].cpu().tolist())

        generated_df = pd.DataFrame.from_dict(df_dic)
        generated_df["event_time"] = generated_df["event_time"].apply(
            lambda x: (
                np.array(x) * (self._data_conf.max_time - self._data_conf.min_time)
            )
            + self._data_conf.min_time
        )

        def truncate_lists(row):
            value = row["trx_count"]
            for col in row.index:
                if isinstance(row[col], (np.ndarray, list)):
                    row[col] = row[col][:value]
            return row

        generated_df = generated_df.apply(func=truncate_lists, axis=1)
        generated_df[self._data_conf.col_id] = np.arange(len(generated_df))
        return generated_df


class GANGenTrainer(GenTrainer):
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        discriminator: torch.nn.Module,
        d_opt: Union[torch.optim.Optimizer, None] = None,
        optimizer: Union[torch.optim.Optimizer, None] = None,
        lr_scheduler: Union[torch.optim.lr_scheduler._LRScheduler, None] = None,
        train_loader: Union[DataLoader, None] = None,
        val_loader: Union[DataLoader, None] = None,
        run_name: Union[str, None] = None,
        total_iters: Union[int, None] = None,
        total_epochs: Union[int, None] = None,
        iters_per_epoch: Union[int, None] = None,
        ckpt_dir: Union[str, os.PathLike, None] = None,
        ckpt_replace: bool = False,
        ckpt_track_metric: str = "epoch",
        ckpt_resume: Union[str, os.PathLike, None] = None,
        device: str = "cpu",
        metrics_on_train: bool = False,
        model_conf: Dict[str, Any] = None,
        data_conf: Dict[str, Any] = None,
        model_conf_d: Dict[str, Any] = None,
    ):
        """Initialize trainer.

        Args:
            model: model to train or validate.
            optimizer: torch optimizer for training.
            lr_scheduler: torch learning rate scheduler.
            train_loader: train dataloader.
            val_loader: val dataloader.
            run_name: for runs differentiation.
            total_iters: total number of iterations to train a model.
            total_epochs: total number of epoch to train a model. Exactly one of
                `total_iters` and `total_epochs` shoud be passed.
            iters_per_epoch: validation and checkpointing are performed every
                `iters_per_epoch` iterations.
            ckpt_dir: path to the directory, where checkpoints are saved.
            ckpt_replace: if `replace` is `True`, only the last and the best checkpoint
                are kept in `ckpt_dir`.
            ckpt_track_metric: if `ckpt_replace` is `True`, the best checkpoint is
                determined based on `track_metric`. All metrcs except loss are assumed
                to be better if the value is higher.
            ckpt_resume: path to the checkpoint to resume training from.
            device: device to train and validate on.
            metrics_on_train: wether to compute metrics on train set.
            model_conf: Model configs from configs/ dir
            data_conf: Data configs from configs/ dir
        """
        assert (total_iters is None) ^ (
            total_epochs is None
        ), "Exactly one of `total_iters` and `total_epochs` shoud be passed."

        self._run_name = (
            run_name if run_name is not None else datetime.now().strftime("%F_%T")
        )

        self._total_iters = total_iters
        self._total_epochs = total_epochs
        self._iters_per_epoch = iters_per_epoch
        self._ckpt_dir = ckpt_dir
        self._ckpt_replace = ckpt_replace
        self._ckpt_track_metric = ckpt_track_metric
        self._ckpt_resume = ckpt_resume
        self._device = device
        self._metrics_on_train = metrics_on_train
        self._model_conf = model_conf
        self._data_conf = data_conf

        self._model = model
        self._model.to(device)
        self._opt = optimizer
        self._sched = lr_scheduler
        self._train_loader = train_loader
        if train_loader is not None:
            self._cyc_train_loader = _CyclicalLoader(train_loader)
        self._val_loader = val_loader

        self._metric_values = None
        self._loss_values = None
        self._last_iter = 0
        self._last_epoch = 0

        self._D = discriminator.to(device)
        self._dopt = d_opt
        self._model_conf_d = model_conf_d

    @property
    def D(self) -> Union[torch.nn.Module, None]:
        return self._D

    def train(self, iters: int) -> None:
        assert self._opt is not None, "Set an optimizer first"
        assert self._train_loader is not None, "Set a train loader first"
        logger.info("Epoch %04d: train started", self._last_epoch + 1)
        self._model.train()
        self._D.train()

        loss_ema = 0.0
        losses: List[float] = []
        d_losses: List[float] = []
        preds, gts = [], []
        pbar = tqdm(zip(range(iters), self._cyc_train_loader), total=iters)
        for i, (inp, gt) in pbar:
            inp, gt = inp.to(self._device), gt.to(self._device)

            ### Generator Step ###
            pred = self._model(inp)
            d_pred = self.D(
                out_to_padded_batch(pred, self._data_conf).to(self.device)
            ).detach()
            d_pred = self.D(
                out_to_padded_batch(pred, self._data_conf).to(self.device)
            ).detach()
            if self._metrics_on_train:
                preds.append(pred)
                gts.append(gt)

            loss = self.compute_loss(pred, gt, d_pred)
            loss.backward()

            loss_np = loss.item()
            losses.append(loss_np)
            loss_ema = loss_np if i == 0 else 0.9 * loss_ema + 0.1 * loss_np
            pbar.set_postfix_str(f"Loss: {loss_ema:.4g}")

            self._opt.step()

            self._last_iter += 1
            self._opt.zero_grad()
            ### DISCRIMINATOR STEP ###
            if ((i + 1) % self._model_conf_d.discriminator_step_every) == 0:
                fake_inp = out_to_padded_batch(pred, self._data_conf).to(self.device)
                d_inp, d_labels = self.mix_batches(fake_inp, inp)
                d_pred = self.D(d_inp)
                d_loss = self.D.loss(d_pred.to(self.device), d_labels)["total_loss"]
                d_loss.backward()

                loss_d = d_loss.item()
                d_losses.append(loss_d)

                self._dopt.step()
                self._dopt.zero_grad()

        self._loss_values = losses
        logger.info(
            "Epoch %04d: avg train loss = %.4g;  avg_D_loss = %.4g",
            self._last_epoch + 1,
            np.mean(losses),
            np.mean(d_losses),
        )

        if self._metrics_on_train:
            self._metric_values = self.compute_metrics(preds, gts)
            logger.info(
                "Epoch %04d: train metrics: %s",
                self._last_epoch + 1,
                str(self._metric_values),
            )

        logger.info("Epoch %04d: train finished", self._last_epoch + 1)

    def mix_batches(self, gen_batch, true_batch):
        new_payload = {}
        for key in gen_batch.payload.keys():
            new_payload[key] = torch.cat(
                [gen_batch.payload[key].detach(), true_batch.payload[key]], dim=0
            )
        new_lens = torch.cat([gen_batch.seq_lens, true_batch.seq_lens - 1])

        new_batch = PaddedBatch(new_payload, new_lens)

        d_labels = torch.zeros(len(new_batch), dtype=torch.long, device=self.device)
        d_labels[: len(new_batch) // 2] = 1
        # true = 0, fake = 1

        return new_batch, d_labels.unsqueeze(0).repeat(2, 1)

    def compute_loss(
        self,
        model_output: Any,
        ground_truth: Tuple[int, int],  # pyright: ignore unused
        d_pred: Any,
    ) -> torch.Tensor:
        """Compute loss for backward.

        The function is called every iteration in training loop to compute loss.

        Args:
            model_output: raw model output as is.
            ground_truth: tuple of raw idx and raw ground truth label from dataloader.
        """
        # assert isinstance(self.model, MegaNet)
        losses = self.model.loss(model_output, ground_truth)
        d_loss = F.softmax(d_pred, dim=1)[:, 1]
        return losses["total_loss"] + d_loss.mean() * self._model_conf.D_weight

    def validate(self) -> None:
        assert self._val_loader is not None, "Set a val loader first"

        logger.info("Epoch %04d: validation started", self._last_epoch + 1)
        self._model.eval()
        loss_dicts = []
        with torch.no_grad():
            for inp, gt in tqdm(self._val_loader):
                inp = inp.to(self._device)
                model_output = self._model(inp)
                d_pred = self.D(
                    out_to_padded_batch(model_output, self._data_conf).to(self.device)
                )
                cur_loss = self._model.loss(model_output, gt)
                cur_loss["D"] = F.softmax(d_pred, dim=1)[:, 1].mean()
                loss_dicts.append(cur_loss)

        self._metric_values = {
            k: np.mean([d[k].item() for d in loss_dicts]) for k in loss_dicts[0]
        }
        logger.info(
            "Epoch %04d: validation metrics: %s",
            self._last_epoch + 1,
            str(self._metric_values),
        )
        logger.info("Epoch %04d: validation finished", self._last_epoch + 1)
