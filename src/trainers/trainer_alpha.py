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
from .trainer_gen import GenTrainer

logger = logging.getLogger("event_seq")


class TrainerAlpha(GenTrainer):
    def predict(
        self, loader: DataLoader, limit: int = 100000000
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

                counter += gt[0].size(0)

                if counter > limit:
                    break

        return preds, gts

    def reconstruct_data(self, train_supervised_loader):
        limit = self._data_conf.get("recon_limit", 50000)

        logger.info(f"Reconstruction started with limit {limit}")
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
        self, loader: DataLoader, limit: int = 100000000
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

                counter += gt[0].size(0)

                if counter > limit:
                    break

        return preds, gts

    def generate_data(self, train_supervised_loader):
        limit = self._data_conf.get("gen_limit", 50000)
        logger.info(f"Generation started with limit {limit}")
        train_out, train_gts = self.generate(train_supervised_loader)
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
