import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Union, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

logger = logging.getLogger("event_seq")


class _CyclicalLoader:
    """Cycles through pytorch dataloader specified number of steps."""

    def __init__(self, base_dataloader):
        self.base_loader = base_dataloader
        self._len = None
        self._iter = iter(self.base_loader)

    def set_iters_per_epoch(self, iters_per_epoch: int):
        self._len = iters_per_epoch

    def __len__(self):
        return self._len

    def __iter__(self):
        self._total_iters = 0
        return self

    def __next__(self):
        assert self._len, "call `set_iters_per_epoch` before use"

        if self._total_iters >= self._len:
            raise StopIteration

        try:
            item = next(self._iter)
        except StopIteration:
            self._iter = iter(self.base_loader)
            item = next(self._iter)
        self._total_iters += 1
        return item


def _grad_norm(named_parameters):
    total_sq_norm = 0.0
    for n, p in named_parameters:
        if p.grad is None:
            print("GRAD IS NONE", n)
        else:
            param_norm = p.grad.detach().data.norm(2)
            total_sq_norm += param_norm.item() ** 2
    return total_sq_norm**0.5


class BaseTrainer:
    """A base class for all trainers."""

    def __init__(
        self,
        *,
        model: nn.Module,
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

    @property
    def model(self) -> Union[nn.Module, None]:
        return self._model

    @property
    def train_loader(self) -> Union[DataLoader, None]:
        return self._train_loader

    @property
    def val_loader(self) -> Union[DataLoader, None]:
        return self._val_loader

    @property
    def optimizer(self) -> Union[torch.optim.Optimizer, None]:
        return self._opt

    @property
    def lr_scheduler(self) -> Union[torch.optim.lr_scheduler._LRScheduler, None]:
        return self._sched

    @property
    def run_name(self):
        return self._run_name

    @property
    def device(self) -> str:
        return self._device

    def _make_key_extractor(self, key):
        def key_extractor(p: Path) -> float:
            metrics = {}
            for it in p.stem.split("_-_"):
                kv = it.split("__")
                assert len(kv) == 2, f"Failed to parse filename: {p.name}"
                k = kv[0]
                v = -float(kv[1]) if ("loss" in k) or ("mse" in k) else float(kv[1])
                metrics[k] = v
            return metrics[key]

        return key_extractor

    def save_ckpt(self, ckpt_path: Union[str, os.PathLike, None] = None) -> None:
        """Save model, optimizer and scheduler states.

        Args:
            ckpt_path: path to checkpoints. If `ckpt_path` is a directory, the
                checkpoint will be saved there with epoch, loss an metrics in the
                filename. All scalar metrics returned from `compute_metrics` are used to
                construct a filename. If full path is specified, the checkpoint will be
                saved exectly there. If `None` `ckpt_dir` from construct is used with
                subfolder named `run_name` from Trainer's constructor.
        """

        if ckpt_path is None and self._ckpt_dir is None:
            logger.warning(
                "`ckpt_path` was not passned to `save_ckpt` and `ckpt_dir` "
                "was not set in Trainer. No checkpoint will be saved."
            )
            return

        if ckpt_path is None:
            assert self._ckpt_dir is not None
            ckpt_path = Path(self._ckpt_dir) / self._run_name

        ckpt_path = Path(ckpt_path)
        ckpt_path.mkdir(parents=True, exist_ok=True)

        ckpt: Dict[str, Any] = {
            "last_iter": self._last_iter,
            "last_epoch": self._last_epoch,
        }
        if self._model:
            ckpt["model"] = self._model.state_dict()
        if self._opt:
            ckpt["opt"] = self._opt.state_dict()
        if self._sched:
            ckpt["sched"] = self._sched.state_dict()

        if not ckpt_path.is_dir():
            torch.save(ckpt, ckpt_path)

        assert self._metric_values
        assert self._loss_values

        metrics = {k: v for k, v in self._metric_values.items() if np.isscalar(v)}
        metrics["loss"] = np.mean(self._loss_values)

        fname = f"epoch__{self._last_epoch:04d}"
        metrics_str = "_-_".join(
            f"{k}__{v:.4g}" for k, v in metrics.items() if k == self._ckpt_track_metric
        )
        # metrics_str = "test"
        if len(metrics_str) > 0:
            fname = "_-_".join((fname, metrics_str))
        fname += ".ckpt"

        torch.save(ckpt, ckpt_path / Path(fname))

        if not self._ckpt_replace:
            return

        all_ckpt = list(ckpt_path.glob("*.ckpt"))
        last_ckpt = max(all_ckpt, key=self._make_key_extractor("epoch"))
        best_ckpt = max(all_ckpt, key=self._make_key_extractor(self._ckpt_track_metric))
        for p in all_ckpt:
            if p != last_ckpt and p != best_ckpt:
                p.unlink()

    def load_ckpt(self, ckpt_fname: Union[str, os.PathLike]) -> None:
        """Load model, optimizer and scheduler states.

        Args:
            ckpt_fname: path to checkpoint.
        """

        ckpt = torch.load(ckpt_fname, map_location=self._device)

        if "model" in ckpt:
            msg = self._model.load_state_dict(ckpt["model"], strict=False)
            print(msg)
        if "opt" in ckpt:
            if self._opt is None:
                logger.warning(
                    "optimizer was not passes, discarding optimizer state "
                    "in the checkpoint"
                )
            else:
                logger.warning(
                    "optimizer is not loaded now due to problems in FineTUning stage"
                )
                # self._opt.load_state_dict(ckpt["opt"])
        if "sched" in ckpt:
            if self._sched is None:
                logger.warning(
                    "scheduler was not passes, discarding scheduler state "
                    "in the checkpoint"
                )
            else:
                logger.warning(
                    "scheduller is not loaded now due to problems in FineTUning stage"
                )
                # self._sched.load_state_dict(ckpt["sched"])
        # self._last_iter = ckpt["last_iter"]
        # self._last_epoch = ckpt["last_epoch"]

    def train(self, iters: int) -> None:
        assert self._opt is not None, "Set an optimizer first"
        assert self._train_loader is not None, "Set a train loader first"

        logger.info("Epoch %04d: train started", self._last_epoch + 1)
        self._model.train()

        loss_ema = 0.0
        losses: List[float] = []
        preds, gts = [], []
        pbar = tqdm(zip(range(iters), self._cyc_train_loader), total=iters)
        pbar.set_description_str(f"Epoch {self._last_epoch + 1: 3}")
        for i, (inp, gt) in pbar:
            inp, gt = inp.to(self._device), gt.to(self._device)

            pred = self._model(inp)
            if self._metrics_on_train:
                preds.append(pred.to("cpu"))
                gts.append(gt.to("cpu"))

            loss = self.compute_loss(pred, gt)
            loss.backward()

            loss_np = loss.item()
            losses.append(loss_np)
            loss_ema = loss_np if i == 0 else 0.9 * loss_ema + 0.1 * loss_np
            pbar.set_postfix_str(f"Loss: {loss_ema:.4g}")

            # CLIP GRADIENTS
            # torch.nn.utils.clip_grad_norm_(self._model.parameters(), 5)
            self._opt.step()

            self._last_iter += 1
            logger.debug(
                "iter: %d,\tloss value: %4g,\tgrad norm: %4g",
                self._last_iter,
                loss.item(),
                _grad_norm(self._model.named_parameters()),
            )

            self._opt.zero_grad()

        self._loss_values = losses
        logger.info(
            "Epoch %04d: avg train loss = %.4g", self._last_epoch + 1, np.mean(losses)
        )

        if self._metrics_on_train:
            self._metric_values = self.compute_metrics(preds, gts)
            logger.info(
                "Epoch %04d: train metrics: %s",
                self._last_epoch + 1,
                str(self._metric_values),
            )

        logger.info("Epoch %04d: train finished", self._last_epoch + 1)

    def validate(self) -> None:
        assert self._val_loader is not None, "Set a val loader first"

        if len(self._val_loader) == 0:
            self._metric_values = {"placeholder": 0}
            return

        logger.info("Epoch %04d: validation started", self._last_epoch + 1)
        preds, gts = self.predict(self._val_loader)

        self._metric_values = self.compute_metrics(preds, gts)
        logger.info(
            "Epoch %04d: validation metrics: %s",
            self._last_epoch + 1,
            str(self._metric_values),
        )
        logger.info("Epoch %04d: validation finished", self._last_epoch + 1)

    def predict(self, loader: DataLoader, limit=None) -> Tuple[List[Any], List[Any]]:
        self._model.eval()
        preds, gts = [], []
        i = 0
        with torch.no_grad():
            for inp, gt in tqdm(loader):
                gts.append(gt.to("cpu"))
                inp = inp.to(self._device)
                pred = self._model(inp)
                preds.append(pred.to("cpu"))
                i += loader.batch_size
                if limit and i > limit:
                    break

        return preds, gts

    def compute_metrics(
        self,
        model_outputs: List[Any],
        ground_truths: List[Any],
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
        raise NotImplementedError

    def compute_loss(
        self,
        model_output: Any,
        ground_truth: Any,
    ) -> torch.Tensor:
        """Compute loss for backward.

        The function is called every iteration in training loop to compute loss.

        Args:
            model_output: raw model output as is.
            ground_truth: raw ground truth label from dataloader.
        """
        raise NotImplementedError

    def log_metrics(
        self,
        phase: Literal["train", "val"],
        metrics: Union[Dict[str, Any], None] = None,
        epoch: Union[int, None] = None,
        losses: Union[List[float], None] = None,
        iterations: Union[List[int], None] = None,
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
        raise NotImplementedError

    def run(self) -> None:
        """Train and validate model."""

        assert self._opt, "Set an optimizer to run full cycle"
        assert self._train_loader is not None, "Set a train loader to run full cycle"
        assert self._val_loader is not None, "Set a val loader to run full cycle"

        logger.info("run %s started", self._run_name)
        logger.info("using following model configs: \n%s", str(self._model_conf))

        if self._ckpt_resume is not None:
            logger.info("Resuming from checkpoint '%s'", str(self._ckpt_resume))
            self.load_ckpt(self._ckpt_resume)

        self._model.to(self._device)

        if self._iters_per_epoch is None:
            logger.warning(
                "`iters_per_epoch` was not passed to the constructor. "
                "Defaulting to the length of the dataloader."
            )
            self._iters_per_epoch = len(self._cyc_train_loader.base_loader)

        if self._total_iters is None:
            assert self._total_epochs is not None
            self._total_iters = self._total_epochs * self._iters_per_epoch

        self._cyc_train_loader.set_iters_per_epoch(self._iters_per_epoch)

        while self._last_iter < self._total_iters:
            train_iters = min(
                self._total_iters - self._last_iter,
                self._iters_per_epoch,
            )
            self.train(train_iters)
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

            self.validate()
            self.log_metrics(
                "val",
                self._metric_values,
                self._last_epoch + 1,
            )

            self._last_epoch += 1
            self.save_ckpt()
            self._metric_values = None
            self._loss_values = None

        logger.info("run '%s' finished successfully", self._run_name)

    def best_checkpoint(self) -> str:
        """
        Return the path to the best checkpoint
        """
        assert self._ckpt_dir is not None
        ckpt_path = Path(self._ckpt_dir) / self._run_name

        ckpt_path = Path(ckpt_path)

        all_ckpt = list(ckpt_path.glob("*.ckpt"))
        best_ckpt = max(all_ckpt, key=self._make_key_extractor(self._ckpt_track_metric))

        return best_ckpt

    def load_best_model(self) -> None:
        """
        Loads the best model to self._model according to the track metric.
        """

        best_ckpt = self.best_checkpoint()
        self.load_ckpt(best_ckpt)

    def test(self, test_loader: DataLoader) -> None:
        """
        Logs test metrics with self.compute_metrics
        """

        logger.info("Test started")
        preds, gts = self.predict(test_loader)

        self._metric_values = self.compute_metrics(preds, gts)
        logger.info(
            "Test metrics: %s",
            str(self._metric_values),
        )
        logger.info("Test finished")

        return self._metric_values
