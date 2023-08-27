from typing import Union, Dict, Any, List, Tuple
import os
from pathlib import Path
import logging
import gc

from tqdm.autonotebook import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


class CyclicalLoader:
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


def _grad_norm(params):
    total_sq_norm = 0.0
    for p in params:
        param_norm = p.grad.detach().data.norm(2)
        total_sq_norm += param_norm.item() ** 2
    return total_sq_norm ** 0.5


class BaseTrainer:
    """A base class for all trainers."""

    def __init__(
        self,
        total_iters: Union[int, None] = None,
        iters_per_epoch: Union[int, None] = None,
        ckpt_dir: Union[str, os.PathLike, None] = None,
        ckpt_replace: bool = False,
        ckpt_track_metric: str = "epoch",
        device: str = "cpu",
    ):
        """Initialize trainer.

        Args:
            total_iters: total number of iterations to train a model.
            iters_per_epoch: validation and checkpointing are performed every
                `iters_per_epoch` iterations.
            ckpt_dir: path to the directory, where checkpoints are saved.
            ckpt_replace: if `replace` is `True`, only the last and the best checkpoint
                are kept in `ckpt_dir`.
            ckpt_track_metric: if `ckpt_replace` is `True`, the best checkpoint is 
                determined based on `track_metric`. All metrcs except loss are assumed
                to be better if the value is higher.
            device: device to train and validate on.
        """
        self._total_iters = total_iters
        self._iters_per_epoch = iters_per_epoch
        self._ckpt_dir = ckpt_dir
        self._ckpt_replace = ckpt_replace
        self._ckpt_track_metric = ckpt_track_metric
        self._device = device

        self.reset()

    def reset(self) -> None:
        """Reset trainer to default state but keep parameters passed to constructor."""

        self._model: Union[nn.Module, None] = None
        self._opt: Union[torch.optim.Optimizer, None] = None
        self._sched: Union[torch.optim.lr_scheduler.LRScheduler, None] = None
        self._train_loader: Union[CyclicalLoader, None] = None
        self._val_loader: Union[CyclicalLoader, None] = None

        self._metric_values = None
        self._loss_values = None
        self._last_iter = 0
        self._last_epoch = 0
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()

    @property
    def model(self) -> Union[nn.Module, None]: return self._model

    @model.setter
    def model(self, model: nn.Module) -> None: self._model = model

    @property
    def train_loader(self) -> Union[DataLoader, None]: 
        if self._train_loader:
            return self._train_loader.base_loader
        return None

    @train_loader.setter
    def train_loader(self, loader: DataLoader) -> None: 
        self._train_loader = CyclicalLoader(loader)

    @property
    def val_loader(self) -> Union[DataLoader, None]: 
        if self._val_loader:
            return self._val_loader.base_loader
        return None

    @val_loader.setter
    def val_loader(self, loader: DataLoader) -> None: 
        self._val_loader = CyclicalLoader(loader)

    @property
    def device(self) -> str: return self._device

    @device.setter
    def device(self, device: str) -> None: self._device = device

    def setup_optim(
        self,
        opt_name: Union[str, None] = None,
        opt_params: Union[Dict, None] = None,
        sched_name: Union[str, None] = None,
        sched_params: Union[Dict, None] = None,
    ) -> None:
        """Setup optimizer and scheduler.

        To enable training at least optimizer shoud be specified.

        Args:
            opt_name: a name of optimizer from `torch.optim`.
            opt_params: keyword arguments that will be passed to the optimizer 
                constructor.
            sched_name: a name of torch scheduler from `torch.optim.lr_scheduler`.
            sched_params: keyword arguments that will be passed to the scheduler
                constructor.
        """
        if opt_name:
            assert hasattr(torch.optim, opt_name), f"Unknown optimizer: {opt_name}"
            assert opt_params
            assert self._model

            opt_class = getattr(torch.optim, opt_name)
            self._opt = opt_class(self._model.parameters(), **opt_params)

        if sched_name:
            assert self._opt
            assert hasattr(torch.optim.lr_scheduler, sched_name),\
                f"Unknown optimizer: {opt_name}"
            assert sched_params

            sched_class = getattr(torch.optim.lr_scheduler, sched_name)
            self._sched = sched_class(self._opt, **sched_params)

    def save_ckpt(self, ckpt_path: Union[str, os.PathLike, None] = None) -> None:
        """Save model, optimizer and scheduler states.

        Args:
            ckpt_path: path to checkpoints. If `ckpt_path` is a directory, the
                checkpoint will be saved there with epoch, loss an metrics in the
                filename. All scalar metrics returned from `compute_metrics` are used to
                construct a filename. If full path is specified, the checkpoint will be 
                saved exectly there. If `None` `ckpt_dir` from construct is used.
        """

        if not ckpt_path:
            ckpt_path = self._ckpt_dir

        if ckpt_path is None:
            return

        ckpt_path = Path(ckpt_path)

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
        metrics["epoch"] = self._last_epoch
        fname = Path(" - ".join(f"{k}: {v:.4e}" for k, v in metrics.items()) + ".cpkt")
        torch.save(ckpt, ckpt_path / fname)

        if not self._ckpt_replace:
            return

        def make_key_extractor(key):
            def key_extractor(p: Path) -> float:
                metrics = {}
                for it in p.stem.split(" - "):
                    kv = it.split(": ")
                    assert len(kv) == 2, f"Failed to parse filename: {p.name}"
                    k = kv[0]
                    v = -float(kv[1]) if k == "loss" else float(kv[1])
                    metrics[k] = v
                return metrics[key]
            return key_extractor

        all_ckpt = ckpt_path.glob("*.ckpt")
        last_ckpt = max(all_ckpt, key=make_key_extractor("epoch"))
        best_ckpt = max(all_ckpt, key=make_key_extractor(self._ckpt_track_metric))
        for p in all_ckpt:
            if p != last_ckpt and p != best_ckpt:
                p.unlink()


    def load_ckpt(self, ckpt_fname: Union[str, os.PathLike]) -> None:
        """Load model, optimizer and scheduler states.

        Args:
            ckpt_fname: path to checkpoint.
        """

        ckpt = torch.load(ckpt_fname)

        if "model" in ckpt:
            assert self._model, "setup a model to load this checkpoint"
            self._model.load_state_dict(ckpt["model"])
        if "opt" in ckpt:
            assert self._opt, "setup an optimizer to load this checkpoint"
            self._opt.load_state_dict(ckpt["opt"])
        if "sched" in ckpt:
            assert self._sched, "setup a scheduler to load this checkpoint"
            self._sched.load_state_dict(ckpt["sched"])
        self._last_iter = ckpt["last_iter"]
        self._last_epoch = ckpt["last_epoch"]

    def train(self, iters: int) -> None:
        assert self._model, "Set a model first"
        assert self._opt, "Set an optimizer first"
        assert self._train_loader, "Set a train loader first"

        logger.info("train")
        self._model.to(self._device)
        self._model.train()
        losses: List[float] = []
        for i, (inp, gt) in tqdm(enumerate(self._train_loader), total=iters):
            inp, gt = inp.to(self._device), gt.to(self._device)

            preds = self._model(inp)
            loss = self.compute_loss(preds, gt)
            loss.backward()
            losses.append(loss.item())

            self._opt.step()
            self._opt.zero_grad()

            self._last_iter += 1
            logger.debug(
                "iter: %d, loss value: %e, grad norm: %e", 
                self._last_iter,
                loss.item(),
                _grad_norm(self._model.parameters()),
            )
            
            if i >= iters - 1:
                break

        self._loss_values = losses
        logger.info("train finished")

    def validate(self) -> Tuple[List[Any], List[Any]]:
        assert self._model, "Set a model first"
        assert self._val_loader, "Set a val loader first"

        logger.info("validation")
        preds, gts = [], []
        with torch.no_grad():
            for inp, gt in tqdm(self._val_loader):
                gts.append(gt)
                inp = inp.to(self._device)
                pred = self._model(preds)
                preds.append(pred.cpu())

        logger.info("validation finished")
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
        ...

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
        ...

    def log_metrics(
        self, 
        metrics: Dict[str, Any], 
        epoch: int,
        losses: List[float],
        iterations: List[int],
    ):
        """Log metrics.

        The metrics are coputed every epoch. The loss is computed every iteration.

        Args:
            metrics: a dict that is returned from `compute_metrics` every epoch.
            epoch: number of epoch after which the metrics were computed.
            losses: a list of loss values.
            iterations: a list of iteration number for corresponding loss values.
        """
        ...

    def run(self) -> None:
        """Train and validate model.

        Args:
        """
        assert self._train_loader, "Set a train loader first"
        assert self._total_iters and self._total_iters > 0,\
            "To run, pass positive `total_iters` to the constructor"

        logger.info("start")

        if self._iters_per_epoch is None:
            logger.warning(
                "`iters_per_epoch` was not passed to the constructor. "
                "Defaulting to the length of the dataloader."
            )
            self._iters_per_epoch = len(self._train_loader.base_loader)

        self._train_loader.set_iters_per_epoch(self._iters_per_epoch)

        while self._last_iter < self._total_iters:

            train_iters = min(
                self._total_iters - self._last_iter, 
                self._iters_per_epoch,
            )
            self.train(train_iters)
            if self._sched:
                self._sched.step()

            preds, gts = self.validate()
            self._metric_values = self.compute_metrics(preds, gts)

            self._last_epoch += 1

            iterations = list(range(
                self._last_iter - self._iters_per_epoch, 
                self._last_iter + 1,
            ))

            assert self._loss_values
            self.log_metrics(
                self._metric_values, 
                self._last_epoch, 
                self._loss_values, 
                iterations
            )
            self.save_ckpt()
