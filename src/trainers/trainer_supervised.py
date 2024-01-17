import logging
from typing import Any, Dict, List, Literal, Union, Tuple

import numpy as np
import torch

from ..models.mTAND.model import MegaNetCE
from .base_trainer import BaseTrainer
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error

logger = logging.getLogger(__name__)


class SimpleTrainerSupervised(BaseTrainer):
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


class AucTrainerSupervised(SimpleTrainerSupervised):
    def compute_metrics(
        self,
        model_outputs: List[Any],
        ground_truths: List[Any],  # pyright: ignore unused
    ) -> Dict[str, Any]:
        loss_dicts = [
            self.model.loss(it, gt) for it, gt in zip(model_outputs, ground_truths)
        ]
        losses_dict = {
            k: np.mean([d[k].item() for d in loss_dicts]) for k in loss_dicts[0]
        }

        preds = torch.cat([it.cpu()[:, 1] for it in model_outputs])
        gold = torch.cat([gt[1].cpu() for gt in ground_truths])
        try:
            score = roc_auc_score(gold, preds)
        except ValueError:
            score = 0.5

        losses_dict["roc_auc"] = score

        return losses_dict


class AccuracyTrainerSupervised(SimpleTrainerSupervised):
    def compute_metrics(
        self,
        model_outputs: List[Any],
        ground_truths: List[Any],  # pyright: ignore unused
    ) -> Dict[str, Any]:
        loss_dicts = [
            self.model.loss(it, gt) for it, gt in zip(model_outputs, ground_truths)
        ]
        losses_dict = {
            k: np.mean([d[k].item() for d in loss_dicts]) for k in loss_dicts[0]
        }

        preds = torch.cat([it.cpu().argmax(dim=1) for it in model_outputs])
        gold = torch.cat([gt[1].cpu() for gt in ground_truths])
        score = accuracy_score(gold, preds)

        losses_dict["accuracy"] = score

        return losses_dict


class MSETrainerSupervised(SimpleTrainerSupervised):
    def compute_metrics(
        self,
        model_outputs: List[Any],
        ground_truths: List[Any],  # pyright: ignore unused
    ) -> Dict[str, Any]:
        loss_dicts = [
            self.model.loss(it, gt) for it, gt in zip(model_outputs, ground_truths)
        ]
        losses_dict = {
            k: np.mean([d[k].item() for d in loss_dicts]) for k in loss_dicts[0]
        }

        preds = torch.cat([it.cpu() for it in model_outputs])
        preds = torch.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
        gold = torch.cat([gt[1].cpu() for gt in ground_truths])

        score = mean_squared_error(gold, preds)

        losses_dict["mse"] = score

        return losses_dict
