import logging
from typing import Any, Dict, List, Literal, Union, Tuple

import numpy as np
import torch
from pathlib import Path
import os
import pandas as pd

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


params = {
    "n_estimators": 200,
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
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
        loss_dicts = [
            self.model.loss(it, gt) for it, gt in zip(model_outputs, ground_truths)
        ]
        losses_dict = {
            k: np.mean([d[k].item() for d in loss_dicts]) for k in loss_dicts[0]
        }

        return losses_dict

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
        self, test_loader: DataLoader, train_supervised_loader: DataLoader
    ) -> None:
        """
        Logs test metrics with self.compute_metrics
        """

        logger.info("Test started")
        train_out, train_gts = self.predict(train_supervised_loader)
        test_out, test_gts = self.predict(test_loader)

        train_embeddings = [out["latent"] for out in train_out]
        test_embeddings = [out["latent"] for out in test_out]

        test_metric = self.compute_test_metric(
            train_embeddings, train_gts, test_embeddings, test_gts
        )
        mean_metric = np.mean(test_metric)
        print(test_metric)
        logger.info("Test metrics: %s, Mean: %s", str(test_metric), str(mean_metric))
        logger.info("Test finished")

        return test_metric

    def compute_test_metric(
        self, train_embeddings, train_gts, test_embeddings, test_gts
    ):
        train_labels = torch.cat([gt[1].cpu() for gt in train_gts]).numpy()
        train_embeddings = torch.cat(train_embeddings).cpu().numpy()

        test_labels = torch.cat([gt[1].cpu() for gt in test_gts]).numpy()
        test_embeddings = torch.cat(test_embeddings).cpu().numpy()

        skf = StratifiedKFold(n_splits=self._model_conf.cv_splits)

        results = []
        for i, (train_index, test_index) in enumerate(
            skf.split(train_embeddings, train_labels)
        ):
            train_emb_subset = train_embeddings[train_index]
            train_labels_subset = train_labels[train_index]

            model = LGBMClassifier(**params)
            preprocessor = MaxAbsScaler()

            train_emb_subset = preprocessor.fit_transform(train_emb_subset)
            test_embeddings_subset = preprocessor.transform(test_embeddings)

            model.fit(train_emb_subset, train_labels_subset)
            y_pred = model.predict_proba(test_embeddings_subset)

            auc_score = roc_auc_score(test_labels, y_pred[:, 1])
            results.append(auc_score)

        return results

    def generate_data(self, train_supervised_loader):
        train_out, train_gts = self.predict(train_supervised_loader)
        generated_data = self.output_to_df(
            train_out, train_gts, self._model_conf, self._data_conf
        )

        gen_data_path = Path(self._ckpt_dir) / "generated_data"
        gen_data_path.mkdir(parents=True, exist_ok=True)
        save_path = gen_data_path / self._run_name

        generated_data.to_parquet(save_path)

        return save_path

    @staticmethod
    def output_to_df(outs, gts, model_conf, conf):
        order = {}

        k = 0
        for key in outs[0]["input_batch"].payload.keys():
            if key in conf.features.numeric_values.keys():
                order[k] = key
                k += 1

        df_dic = {"event_time": [], "trx_count": [], "target_target_flag": []}
        for feature in conf.features.embeddings.keys():
            df_dic[feature] = []

        for feature in conf.features.numeric_values.keys():
            df_dic[feature] = []

        for out, gt in zip(outs, gts):
            for key, val in out["emb_dist"].items():
                df_dic[key].extend((val.cpu().argmax(dim=-1) - 1).tolist())

            if model_conf.use_deltas:
                pred_delta = out["pred"][:, :, -1].squeeze(-1)
                pred = out["pred"][:, :, :-1]

            num_numeric = len(conf.features.numeric_values.keys())
            numeric_pred = pred[:, :, -num_numeric:]
            for i in range(num_numeric):
                cur_key = order[i]
                cur_val = numeric_pred[:, :, i].cpu().tolist()
                df_dic[cur_key].extend(cur_val)

            df_dic["event_time"].extend(out["time_steps"][:, 1:].tolist())
            df_dic["trx_count"].extend(
                (out["time_steps"][:, 1:] != -1).sum(dim=1).tolist()
            )
            df_dic["target_target_flag"].extend(gt[1].cpu().tolist())

        generated_df = pd.DataFrame.from_dict(df_dic)
        generated_df["event_time"] = generated_df["event_time"].apply(
            lambda x: (np.array(x) * (conf.max_time - conf.min_time)) + conf.min_time
        )

        def truncate_lists(row):
            value = row["trx_count"]
            for col in row.index:
                if isinstance(row[col], (np.ndarray, list)):
                    row[col] = row[col][:value]
            return row

        generated_df = generated_df.apply(func=truncate_lists, axis=1)
        generated_df[conf.col_id] = np.arange(len(generated_df))
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
            if self._metrics_on_train:
                preds.append(pred)
                gts.append(gt)

            loss = self.compute_loss(pred, gt)
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
                fake_inp = self.out_to_padded_batch(pred)
                d_inp, d_labels = self.mix_batches(fake_inp, inp)
                d_pred = self.D(d_inp)
                d_loss = self.D.loss(d_pred, d_labels)
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

    def out_to_padded_batch(self, out):
        order = {}

        k = 0
        for key in out["input_batch"].payload.keys():
            if key in self._data_conf.features.numeric_values.keys():
                order[k] = key
                k += 1

        num_numeric = len(self._data_conf.features.numeric_values.keys())

        payload = {}
        payload["event_time"] = out["time_steps"][:, 1:]
        length = (out["time_steps"][:, 1:] != -1).sum(dim=1)
        mask = out["time_steps"][:, 1:] != -1
        for key, val in out["emb_dist"].items():
            payload[key] = val.cpu().argmax(dim=-1).detach()
            payload[key][~mask] = 0

        if self._model_conf.use_deltas:
            pred_delta = out["pred"][:, :, -1].squeeze(-1)
            pred = out["pred"][:, :, :-1]

        numeric_pred = pred[:, :, -num_numeric:]
        for i in range(num_numeric):
            cur_key = order[i]
            cur_val = numeric_pred[:, :, i].cpu().detach()
            payload[cur_key] = cur_val
            payload[cur_key][~mask] = 0

        return PaddedBatch(payload, length)

    @staticmethod
    def mix_batches(gen_batch, true_batch):
        new_payload = {}
        for key in gen_batch.payload.keys():
            new_payload[key] = torch.cat(
                [gen_batch.payload[key], true_batch.payload[key][:, :-1]], dim=0
            )
        new_lens = torch.cat([gen_batch.seq_lens, true_batch.seq_lens - 1])

        new_batch = PaddedBatch(new_payload, new_lens)

        d_labels = torch.zeros(len(new_batch), dtype=torch.long)
        d_labels[: len(new_batch) // 2] = 1

        return new_batch, d_labels.unsqueeze(0).repeat(2, 1)
