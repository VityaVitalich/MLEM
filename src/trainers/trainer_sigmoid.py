import logging
from typing import Any, Dict, List, Literal, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import os
from copy import deepcopy
import pandas as pd

from .trainer_gen import GenTrainer
from ..data_load.dataloader import PaddedBatch

from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm


logger = logging.getLogger("event_seq")


class SigmoidTrainer(GenTrainer):
    def __init__(
        self,
        *,
        model: nn.Module,
        contrastive_model: nn.Module,
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
        super().__init__(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            run_name=run_name,
            total_iters=total_iters,
            total_epochs=total_epochs,
            iters_per_epoch=iters_per_epoch,
            ckpt_dir=ckpt_dir,
            ckpt_replace=ckpt_replace,
            ckpt_track_metric=ckpt_track_metric,
            ckpt_resume=ckpt_resume,
            device=device,
            metrics_on_train=metrics_on_train,
            model_conf=model_conf,
            data_conf=data_conf,
        )
        self.contrastive_model = contrastive_model

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
        # Reconstruction Loss
        losses = self.model.loss(model_output, ground_truth)
        # Contrastive component
       # loss_sigmoid = torch.tensor(0, device='cuda') 
        loss_sigmoid = self.sigmoid_loss(model_output)
        losses["contrastive"] = loss_sigmoid
        losses["total_loss"] = (
            self._model_conf.reconstruction_weight * losses["total_loss"]
            + self._model_conf.contrastive_weight * loss_sigmoid
        )
        return losses["total_loss"]

    def sigmoid_loss(self, model_output):
        # https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py
        # https://arxiv.org/abs/2303.15343
        lens = model_output["gt"]["input_batch"].seq_lens
        input_batch = PaddedBatch({k: v.detach().clone() for k, v in model_output["gt"]["input_batch"].payload.items()}, lens)
        with torch.no_grad():
            contrastive_output = self.contrastive_model(
                input_batch
            ).detach()

        if ('rosbank' in str(self._data_conf.train_path)) or ('physionet' in str(self._data_conf.train_path)):
            z_recon = model_output['latent']
            z_contrastive = contrastive_output
        else:
            z_recon = F.normalize(model_output["latent"])
            z_contrastive = F.normalize(contrastive_output)
       

        logits = (z_recon @ z_contrastive.T) * self.temp + self.bias
        m1_diag1 = -torch.ones_like(logits) + 2 * torch.eye(logits.size(0)).to(
            logits.device
        )
        loglik = F.logsigmoid(m1_diag1 * logits)
        nll = -torch.sum(loglik, axis=-1)
        return nll.mean()

    @property
    def temp(self):
        return torch.exp(self.model.sigmoid_temp)

    @property
    def bias(self):
        return self.model.sigmoid_bias

    def validate(self) -> None:
        assert self._val_loader is not None, "Set a val loader first"

        logger.info("Epoch %04d: validation started", self._last_epoch + 1)
        self._model.eval()
        loss_dicts = []
        with torch.no_grad():
            for inp, gt in tqdm(self._val_loader):
                inp = inp.to(self._device)
                model_output = self._model(inp)
                loss_dicts.append(self.compute_all_losses(model_output, gt))

        self._metric_values = {
            k: np.mean([d[k].item() for d in loss_dicts]) for k in loss_dicts[0]
        }
        logger.info(
            "Epoch %04d: validation metrics: %s",
            self._last_epoch + 1,
            str(self._metric_values),
        )
        logger.info("Epoch %04d: validation finished", self._last_epoch + 1)

    def compute_all_losses(
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
        # Reconstruction Loss
        losses = self.model.loss(model_output, ground_truth)
        # Contrastive component
       # loss_sigmoid = torch.tensor(0, device='cuda') 
        loss_sigmoid = self.sigmoid_loss(model_output)
        losses["contrastive"] = loss_sigmoid
        losses["total_loss"] = (
            self._model_conf.reconstruction_weight * losses["total_loss"]
            + self._model_conf.contrastive_weight * loss_sigmoid
        )
        return losses