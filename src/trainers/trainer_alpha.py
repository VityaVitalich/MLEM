import logging
from typing import Any, Dict, List, Literal, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import os
import pandas as pd
import math

import numpy as np
import scipy
from sklearn.linear_model import LinearRegression

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
)
from .trainer_gen import GenTrainer
from .trainer_timegan import TGTrainer
from sklearn.linear_model import LogisticRegression


def calc_intrinsic_dimension(other_embeddings):
    all_embeddings = []
    for other_embedding in other_embeddings:
        if other_embedding is not None:
            all_embeddings.append(torch.cat(other_embedding).cpu())

    X = torch.cat(all_embeddings, dim=0).numpy()

    N = X.shape[0]

    dist = scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(X, metric="euclidean")
    )

    # FOR EACH POINT, COMPUTE mu_i = r_2 / r_1,
    # where r_1 and r_2 are first and second shortest distances
    mu = np.zeros(N)

    for i in range(N):
        sort_idx = np.argsort(dist[i, :])
        mu[i] = dist[i, sort_idx[2]] / (dist[i, sort_idx[1]] + 1e-15)

    # COMPUTE EMPIRICAL CUMULATE
    sort_idx = np.argsort(mu)
    Femp = np.arange(N) / N

    # FIT (log(mu_i), -log(1-F(mu_i))) WITH A STRAIGHT LINE THROUGH ORIGIN
    lr = LinearRegression(fit_intercept=False)
    features = np.log(mu[sort_idx]).reshape(-1, 1)
    features = np.clip(features, 1e-15, 1e15)
    lr.fit(features, -np.log(1 - Femp).reshape(-1, 1))

    d = lr.coef_[0][0]  # extract slope

    return d


def calc_anisotropy(other_embeddings):
    all_embeddings = []
    for other_embedding in other_embeddings:
        if other_embedding is not None:
            all_embeddings.append(torch.cat(other_embedding).cpu())

    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    num_embds = min(all_embeddings.size(1), 20001) - 1
    all_embeddings = all_embeddings[:num_embds,:]


    U, S, Vt = torch.linalg.svd(all_embeddings, full_matrices=False)

    return S[0] / S.sum()


logger = logging.getLogger("event_seq")


class TrainerAlpha(GenTrainer):

    def test(
        self, train_supervised_loader: DataLoader, other_loaders: List[DataLoader]
    ) -> Dict[str, float]:
        """
        Logs test metrics with self.compute_metrics
        """
        logger.info("Test started")
        train_out, train_gts = self.get_embeddings(train_supervised_loader)
        other_outs, other_gts = [], []
        for other_loader in other_loaders:
            other_out, other_gt = (
                self.get_embeddings(other_loader) if len(other_loader) > 0 else (None, None)
            )
            other_outs.append(other_out), other_gts.append(other_gt)

        train_embeddings = train_out#[out["latent"] for out in train_out]
        other_embeddings = [
            other_out if other_out is not None else None
            for other_out in other_outs
        ]
        anisotropy = calc_anisotropy(other_embeddings).item()
        logger.info("Anisotropy: %s", str(anisotropy))

        intrinsic_dimension = calc_intrinsic_dimension(
           other_embeddings
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



        def get_metric(model, x, target):
            if metric == "roc_auc":
                return roc_auc_score(target, model.predict_proba(x)[:, 1])
            elif metric == "accuracy":
                return accuracy_score(target, model.predict(x))
            else:
                raise NotImplementedError(f"Unknown objective {metric}")

        model = LogisticRegression()
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

    def get_embeddings(self, loader):

        self._model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for inp, gt in tqdm(loader):
                gts.append(gt.to(self._device))
                inp = inp.to(self._device)
                out = self._model(inp)
                out = self.dict_to_cpu(out)

                preds.append(out['latent'])

        return preds, gts
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

class AlphaTGTrainer(TrainerAlpha, TGTrainer):

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
                out = self._model.reconstruct(inp)
                out = self.dict_to_cpu(out)
                preds.append(out)

                counter += gt[0].size(0)

                if counter > limit:
                    break

        return preds, gts
    
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
                out = self._model.generate(inp)
                out = self.dict_to_cpu(out)
                preds.append(out)

                counter += gt[0].size(0)

                if counter > limit:
                    break

        return preds, gts