import logging
from typing import Any, Dict, List, Literal, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import os
import pandas as pd
import torch.nn as nn

from ..models.mTAND.model import MegaNetCE
from ..data_load.dataloader import PaddedBatch
from .base_trainer import BaseTrainer, _CyclicalLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from lightgbm import LGBMClassifier
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


params = {
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
    "n_jobs": 8,
    "reg_alpha": None,
    "reg_lambda": None,
    "colsample_bytree": None,
    "min_child_samples": None,
    "verbosity": -1,
}


logger = logging.getLogger("event_seq")


class TGTrainer(BaseTrainer):
    def __init__(
        self,
        *,
        model: nn.Module,
        lr_scheduler: Union[torch.optim.lr_scheduler._LRScheduler, None] = None,
        train_loader: Union[DataLoader, None] = None,
        val_loader: Union[DataLoader, None] = None,
        run_name: Union[str, None] = None,
        total_iters: Union[int, None] = None,
        total_epochs_recon: Union[int, None] = None,
        total_epochs_gen: Union[int, None] = None,
        total_epochs_joint: Union[int, None] = None,
        iters_per_epoch: Union[int, None] = None,
        ckpt_dir: Union[str, os.PathLike, None] = None,
        ckpt_replace: bool = False,
        ckpt_track_metric: str = "epoch",
        ckpt_resume: Union[str, os.PathLike, None] = None,
        device: str = "cpu",
        metrics_on_train: bool = False,
        model_conf: Dict[str, Any] = None,
        data_conf: Dict[str, Any] = None,
        optimizer_e: Union[torch.optim.Optimizer, None] = None,
        optimizer_d: Union[torch.optim.Optimizer, None] = None,
        optimizer_g: Union[torch.optim.Optimizer, None] = None,
    ):
        self._run_name = (
            run_name if run_name is not None else datetime.now().strftime("%F_%T")
        )

        self._total_iters = total_iters
        self._total_epochs_recon = total_epochs_recon
        self._total_epochs_gen = total_epochs_gen
        self._total_epochs_joint = total_epochs_joint
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
        self._opt_e = optimizer_e
        self._opt_d = optimizer_d
        self._opt_g = optimizer_g

        self._sched = lr_scheduler
        self._train_loader = train_loader
        if train_loader is not None:
            self._cyc_train_loader = _CyclicalLoader(train_loader)
        self._val_loader = val_loader

        self._metric_values = None
        self._loss_values = None
        self._last_iter = 0
        self._last_epoch = 0

        ckpt_path = Path(self._ckpt_dir) / self._run_name
        ckpt_path.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """Train and validate model."""

        assert self._train_loader is not None, "Set a train loader to run full cycle"
        assert self._val_loader is not None, "Set a val loader to run full cycle"

        logger.info("Embedding run %s started", self._run_name)
        logger.info("using following model configs: \n%s", str(self._model_conf))

        self._model.to(self._device)

        if self._iters_per_epoch is None:
            logger.warning(
                "`iters_per_epoch` was not passed to the constructor. "
                "Defaulting to the length of the dataloader."
            )
            self._iters_per_epoch = len(self._cyc_train_loader.base_loader)

        if self._total_iters is None:
            assert self._total_epochs_recon is not None
            self._total_iters_recon = self._total_epochs_recon * self._iters_per_epoch
            self._total_iters_gen = self._total_epochs_gen * self._iters_per_epoch
            self._total_iters_joint = self._total_epochs_joint * self._iters_per_epoch

        self._cyc_train_loader.set_iters_per_epoch(self._iters_per_epoch)

        self.perform_run(self.train_emb, self.validate_emb, self._total_iters_recon)
        logger.info("Generative run %s started", self._run_name)
        self.perform_run(self.train_gen, self.validate_gen, self._total_iters_gen)
        logger.info("Joint run %s started", self._run_name)
        self.perform_run(self.train_joint, self.validate_joint, self._total_iters_gen)

        logger.info("run '%s' finished successfully", self._run_name)

    def perform_run(self, train_func, validate_func, total_iters):
        self._last_iter = 0
        self._last_epoch = 0

        while self._last_iter < total_iters:
            train_iters = min(
                total_iters - self._last_iter,
                self._iters_per_epoch,
            )
            train_func(train_iters)
            if self._sched:
                self._sched.step()

            iterations = list(
                range(
                    self._last_iter - self._iters_per_epoch,
                    self._last_iter + 1,
                )
            )

            self.log_metrics(
                "train",
                self._metric_values,
                self._last_epoch + 1,
                self._loss_values,
                iterations,
            )
            self._metric_values = None

            validate_func()
            self.log_metrics(
                "val",
                self._metric_values,
                self._last_epoch + 1,
            )

            self._last_epoch += 1
            #  self.save_ckpt() непонятно как сохранять по каким лоссам и тд
            self._metric_values = None
            self._loss_values = None

    def train_emb(self, iters: int) -> None:
        assert self._train_loader is not None, "Set a train loader first"
        logger.info("Epoch %04d: train started", self._last_epoch + 1)
        self._model.train()

        loss_ema = 0.0
        losses: List[float] = []
        preds, gts = [], []
        pbar = tqdm(zip(range(iters), self._cyc_train_loader), total=iters)
        for i, (inp, gt) in pbar:
            inp, gt = inp.to(self._device), gt.to(self._device)

            ### Generator Step ###
            global_hidden, loss = self._model.train_embedder(inp)
            loss.backward()

            loss_np = loss.item()
            losses.append(loss_np)
            loss_ema = loss_np if i == 0 else 0.9 * loss_ema + 0.1 * loss_np
            pbar.set_postfix_str(f"Loss: {loss_ema:.4g}")

            self._opt_e.step()

            self._last_iter += 1
            self._opt_e.zero_grad()

        self._loss_values = losses
        logger.info(
            "Epoch %04d: avg train loss = %.4g;",
            self._last_epoch + 1,
            np.mean(losses),
        )

        logger.info("Epoch %04d: train finished", self._last_epoch + 1)

    def train_gen(self, iters: int) -> None:
        assert self._train_loader is not None, "Set a train loader first"
        logger.info("Epoch %04d: train started", self._last_epoch + 1)
        self._model.train()

        loss_ema = 0.0
        losses: List[float] = []
        preds, gts = [], []
        pbar = tqdm(zip(range(iters), self._cyc_train_loader), total=iters)
        for i, (inp, gt) in pbar:
            inp, gt = inp.to(self._device), gt.to(self._device)

            loss = self._model.train_generator(inp)
            loss.backward()

            loss_np = loss.item()
            losses.append(loss_np)
            loss_ema = loss_np if i == 0 else 0.9 * loss_ema + 0.1 * loss_np
            pbar.set_postfix_str(f"Loss: {loss_ema:.4g}")

            self._opt_g.step()

            self._last_iter += 1
            self._opt_g.zero_grad()

        self._loss_values = losses
        logger.info(
            "Epoch %04d: avg train loss = %.4g;",
            self._last_epoch + 1,
            np.mean(losses),
        )

        logger.info("Epoch %04d: train finished", self._last_epoch + 1)

    def train_joint(self, iters: int) -> None:
        assert self._train_loader is not None, "Set a train loader first"
        logger.info("Epoch %04d: train started", self._last_epoch + 1)
        self._model.train()

        losses_e: List[float] = []
        losses_g: List[float] = []
        losses_d: List[float] = []
        pbar = tqdm(zip(range(iters), self._cyc_train_loader), total=iters)
        for i, (inp, gt) in pbar:
            inp, gt = inp.to(self._device), gt.to(self._device)

            g_loss_u, g_loss_s, g_loss_v, e_loss = self._model.train_joint(inp)
            g_loss_u.backward(retain_graph=True)
            g_loss_s.backward(retain_graph=True)
            g_loss_v.backward(retain_graph=True)
            e_loss.backward()

            loss_np = g_loss_u.item() + g_loss_v.item() + g_loss_s.item()
            losses_g.append(loss_np)

            loss_np = e_loss.item()
            losses_e.append(loss_np)
            pbar.set_postfix_str(f"Loss: {loss_np:.4g}")

            self._opt_g.step()
            self._opt_e.step()

            self._last_iter += 1
            self._opt_g.zero_grad()
            self._opt_e.zero_grad()

            if (i + 1) % 2 == 0:
                d_loss = self._model.train_discriminator(inp)
                d_loss.backward()

                loss_np = d_loss.item()
                losses_d.append(loss_np)

                self._opt_d.step()
                self._opt_d.zero_grad()

        self._loss_values = losses_e
        logger.info(
            "Epoch %04d: avg train embeddder loss = %.4g; generative loss = %.4g; D loss = %.4g;",
            self._last_epoch + 1,
            np.mean(losses_e),
            np.mean(losses_g),
            np.mean(losses_d),
        )

        logger.info("Epoch %04d: train finished", self._last_epoch + 1)

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
        train_out, train_gts = self.predict(train_supervised_loader)
        other_outs, other_gts = [], []
        for other_loader in other_loaders:
            other_out, other_gt = (
                self.predict(other_loader) if len(other_loader) > 0 else (None, None)
            )
            other_outs.append(other_out), other_gts.append(other_gt)

        train_embeddings = [out[0] for out in train_out]
        other_embeddings = [
            [out[0] for out in other_out] if other_out is not None else None
            for other_out in other_outs
        ]

        anisotropy = calc_anisotropy(train_embeddings, other_embeddings).item()
        logger.info("Anisotropy: %s", str(anisotropy))

        intrinsic_dimension = calc_intrinsic_dimension(
            train_embeddings, other_embeddings
        )
        logger.info("Intrinsic Dimension: %s", str(intrinsic_dimension))

        train_metric, other_metrics = self.compute_test_metric(
            train_embeddings, train_gts, other_embeddings, other_gts
        )
        logger.info("Train metrics: %s", str(train_metric))
        logger.info("Other metrics: %s", str(other_metrics))
        logger.info("Test finished")

        return train_metric, other_metrics

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

        if metric == "roc_auc":
            params["objective"] = "binary"
            params["metric"] = "auc"
        elif metric == "accuracy":
            params["objective"] = "multiclass"
        else:
            raise NotImplementedError(f"Unknown objective {metric}")

        def get_metric(model, x, target):
            if metric == "roc_auc":
                return roc_auc_score(target, model.predict_proba(x)[:, 1])
            elif metric == "accuracy":
                return accuracy_score(target, model.predict(x))
            else:
                raise NotImplementedError(f"Unknown objective {metric}")

        model = LGBMClassifier(
            **params,
        )
        preprocessor = MaxAbsScaler()
        train_embeddings_tr = preprocessor.fit_transform(train_embeddings)
        model.fit(train_embeddings_tr, train_labels)

        train_metric = get_metric(model, train_embeddings_tr, train_labels)
        other_metrics = []
        for i, (other_embedding, other_label) in enumerate(
            zip(other_embeddings_new, other_labels)
        ):
            if other_embedding is not None:
                other_embedding_proccesed = preprocessor.transform(other_embedding)
                other_metrics.append(
                    get_metric(model, other_embedding_proccesed, other_label)
                )
            else:
                other_metrics.append(0)
        return train_metric, other_metrics

    def predict(self, loader: DataLoader) -> Tuple[List[Any], List[Any]]:
        self._model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for inp, gt in tqdm(loader):
                gts.append(gt.to(self._device))
                inp = inp.to(self._device)
                out = self._model.train_embedder(inp)
                preds.append(out)

        return preds, gts

    def validate_emb(self) -> None:
        assert self._val_loader is not None, "Set a val loader first"

        logger.info("Epoch %04d: validation started", self._last_epoch + 1)
        self._model.eval()
        losses = []
        with torch.no_grad():
            for inp, gt in tqdm(self._val_loader):
                inp = inp.to(self._device)
                global_hidden, loss = self._model.train_embedder(inp)
                losses.append(loss.item())

        self._metric_values = np.mean(losses)
        logger.info(
            "Epoch %04d: validation metrics: %s",
            self._last_epoch + 1,
            str(self._metric_values),
        )
        logger.info("Epoch %04d: validation finished", self._last_epoch + 1)

    def validate_gen(self) -> None:
        assert self._val_loader is not None, "Set a val loader first"

        logger.info("Epoch %04d: validation started", self._last_epoch + 1)
        self._model.eval()
        losses = []
        with torch.no_grad():
            for inp, gt in tqdm(self._val_loader):
                inp = inp.to(self._device)
                loss = self._model.train_generator(inp)
                losses.append(loss.item())

        self._metric_values = np.mean(losses)
        logger.info(
            "Epoch %04d: validation metrics: %s",
            self._last_epoch + 1,
            str(self._metric_values),
        )
        logger.info("Epoch %04d: validation finished", self._last_epoch + 1)

    def validate_joint(self) -> None:
        assert self._val_loader is not None, "Set a val loader first"

        logger.info("Epoch %04d: validation started", self._last_epoch + 1)
        self._model.eval()
        losses_e = []
        losses_g = []
        with torch.no_grad():
            for inp, gt in tqdm(self._val_loader):
                inp = inp.to(self._device)
                g_loss_u, g_loss_s, g_loss_v, e_loss = self._model.train_joint(inp)
                losses_g.append(g_loss_u.item() + g_loss_s.item() + g_loss_v.item())
                losses_e.append(e_loss.item())

        self._metric_values = [np.mean(losses_g), np.mean(losses_e)]
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
                out[k] = TGTrainer.dict_to_cpu(val)
            else:
                out[k] = val.to("cpu")

        return out

    def reconstruct(self, loader):
        self._model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for inp, gt in tqdm(loader):
                gts.append(gt.to(self._device))
                inp = inp.to(self._device)
                out = self._model.reconstruct(inp)
                out = self.dict_to_cpu(out)
                preds.append(out)

        return preds, gts

    def reconstruct_data(self, train_supervised_loader):
        logger.info("Reconstruction started")
        train_out, train_gts = self.reconstruct(train_supervised_loader)
        logger.info("Reconstructions convertation started")

        reconstructed_data = self.output_to_df(train_out, train_gts)
        logger.info("Reconstructions converted")
        save_path = (
            Path(self._ckpt_dir) / f"{self._run_name}" / "reconstructed_data.parquet"
        )

        reconstructed_data.to_parquet(save_path)
        logger.info("Reconstructions saved")
        return save_path

    def generate(self, loader: DataLoader) -> Tuple[List[Any], List[Any]]:
        self._model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for inp, gt in tqdm(loader):
                gts.append(gt.to(self._device))
                inp = inp.to(self._device)
                out = self._model.generate(inp)
                out = self.dict_to_cpu(out)
                preds.append(out)

        return preds, gts

    def generate_data(self, train_supervised_loader):
        logger.info("Generation started")
        train_out, train_gts = self.generate(train_supervised_loader)
        logger.info("Predictions convertation started")

        generated_data = self.output_to_df(train_out, train_gts)
        logger.info("Predictions converted")
        save_path = (
            Path(self._ckpt_dir) / f"{self._run_name}" / "generated_data.parquet"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)

        generated_data.to_parquet(save_path)
        logger.info("Predictions saved")
        return save_path

    def output_to_df(self, outs, gts):
        df_dic = {
            "event_time": [],
            "trx_count": [],
            self._data_conf.features.target_col: [],
        }
        for feature in self._data_conf.features.embeddings.keys():
            df_dic[feature] = []

        for feature in self._data_conf.features.numeric_values.keys():
            df_dic[feature] = []

        for out, gt in zip(outs, gts):
            for key, val in out["pred"].items():
                if key in self._data_conf.features.embeddings.keys():
                    df_dic[key].extend((val.cpu().argmax(dim=-1) - 1).tolist())
                elif key in self._data_conf.features.numeric_values.keys():
                    df_dic[key].extend(val.cpu().squeeze(-1).tolist())

            if "delta" in out["pred"].keys():
                pred_delta = out["pred"]["delta"].cumsum(1)
                df_dic["event_time"].extend(pred_delta.tolist())
                df_dic["trx_count"].extend((pred_delta != -1).sum(dim=1).tolist())
            else:
                df_dic["event_time"].extend(out["time_steps"].tolist())
                df_dic["trx_count"].extend(
                    (out["time_steps"] != -1).sum(dim=1).tolist()
                )

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
