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
from .trainer_gen import GenTrainer
from .trainer_alpha import TrainerAlpha


class TrainerDDPM(GenTrainer):
    def predict(self, loader: DataLoader) -> Tuple[List[Any], List[Any]]:
        self._model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for inp, gt in tqdm(loader):
                gts.append(gt.to(self._device))
                inp = inp.to(self._device)
                out = self._model(inp, need_delta=True)
                out = self.dict_to_cpu(out)

                out["gt"].pop("input_batch")
                out.pop("all_latents", None)
                preds.append(out)

        return preds, gts


class TrainerAlphaDDPM(TrainerAlpha):
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
                out = self._model(inp, need_delta=True)
                out = self.dict_to_cpu(out)

                out["gt"].pop("input_batch")
                out.pop("all_latents", None)
                preds.append(out)

                counter += gt[0].size(0)

                if counter > limit:
                    break

        return preds, gts
