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
)
from .trainer_gen import GenTrainer
from .trainer_timegan import TGTrainer
from sklearn.linear_model import LogisticRegression, LinearRegression


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



def calc_intrinsic_dimension(other_embeddings):
    all_embeddings = []
    for other_embedding in other_embeddings:
        if other_embedding is not None:
            all_embeddings.append(torch.cat(other_embedding).cpu())

    X = torch.cat(all_embeddings, dim=0).numpy()[:20000,]

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
    num_embds = min(all_embeddings.size(0), 10001) - 1
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
        predict_limit = self._data_conf.get("predict_limit", 100000)
        train_out, train_gts = self.get_embeddings(train_supervised_loader, predict_limit)
        other_outs, other_gts = [], []
        for other_loader in other_loaders:
            other_out, other_gt = (
                self.get_embeddings(other_loader, predict_limit) if len(other_loader) > 0 else (None, None)
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

        train_metric, other_metrics, lin_prob_metrics = self.compute_test_metric(
            train_embeddings, train_gts, other_embeddings, other_gts
        )
        logger.info("Train metrics: %s", str(train_metric))
        logger.info("Validation, supervised Test, Fixed Test Metrics: %s", str(other_metrics))
        logger.info("LinProb Validation, supervised Test, Fixed Test Metrics: %s", str(lin_prob_metrics))

        logger.info("Test finished")
        return train_metric, other_metrics, lin_prob_metrics
    
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
            model = LGBMClassifier(
                **params,
            )
            lin_prob = LogisticRegression(max_iter=5000)
        elif metric == "accuracy":
            params["objective"] = "multiclass"
            model = LGBMClassifier(
                **params,
            )
            lin_prob = LogisticRegression(max_iter=5000)
        elif metric == 'mse':
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
            elif metric == 'mse':
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
                lin_prob_metrics.append(get_metric(lin_prob, other_embedding_proccesed, other_label))
            else:
                other_metrics.append(0)
                lin_prob_metrics.append(0)
        
        return train_metric, other_metrics, lin_prob_metrics

    def get_embeddings(self, loader, limit: int = 50000):
        counter = 0
        self._model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for inp, gt in tqdm(loader):
                gts.append(gt.to(self._device))
                inp = inp.to(self._device)
                out = self._model(inp)
                out = self.dict_to_cpu(out)

                preds.append(out['latent'])

                counter += gt[0].size(0)
                if counter > limit:
                    break

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

        for out, gt in zip(outs, gts):
            for key, val in out["pred"].items():
                if key in self._data_conf.features.embeddings.keys():
                    df_dic[key].extend((val.cpu().argmax(dim=-1)).tolist())
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


class AlphaTGTrainer(TGTrainer):


    def test(
        self, train_supervised_loader: DataLoader, other_loaders: List[DataLoader]
    ) -> Dict[str, float]:
        """
        Logs test metrics with self.compute_metrics
        """
        logger.info("Test started")
        predict_limit = self._data_conf.get("predict_limit", 100000)
        train_out, train_gts = self.get_embeddings(train_supervised_loader, predict_limit)
        other_outs, other_gts = [], []
        for other_loader in other_loaders:
            other_out, other_gt = (
                self.get_embeddings(other_loader, predict_limit) if len(other_loader) > 0 else (None, None)
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

        train_metric, other_metrics, lin_prob_metrics = self.compute_test_metric(
            train_embeddings, train_gts, other_embeddings, other_gts
        )
        logger.info("Train metrics: %s", str(train_metric))
        logger.info("Validation, supervised Test, Fixed Test Metrics: %s", str(other_metrics))
        logger.info("LinProb Validation, supervised Test, Fixed Test Metrics: %s", str(lin_prob_metrics))

        logger.info("Test finished")
        return train_metric, other_metrics, lin_prob_metrics


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
            model = LGBMClassifier(
                **params,
            )
            lin_prob = LogisticRegression(max_iter=5000)
        elif metric == "accuracy":
            params["objective"] = "multiclass"
            model = LGBMClassifier(
                **params,
            )
            lin_prob = LogisticRegression(max_iter=5000)
        elif metric == 'mse':
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
            elif metric == 'mse':
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
                lin_prob_metrics.append(get_metric(lin_prob, other_embedding_proccesed, other_label))
            else:
                other_metrics.append(0)
                lin_prob_metrics.append(0)
        
        return train_metric, other_metrics, lin_prob_metrics

    def get_embeddings(self, loader, limit: int = 50000):
        counter = 0
        self._model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for inp, gt in tqdm(loader):
                gts.append(gt.to(self._device))
                inp = inp.to(self._device)
                global_hidden, loss = self._model.train_embedder(inp)
                #out = self.dict_to_cpu(out)

                preds.append(global_hidden.to('cpu'))

                counter += gt[0].size(0)
                if counter > limit:
                    break

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

    def output_to_df(self, outs, gts, **kwargs):
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
                    df_dic[key].extend((val.cpu().argmax(dim=-1)).tolist())
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
