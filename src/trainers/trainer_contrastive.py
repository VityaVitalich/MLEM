import logging
from typing import Any, Dict, List, Literal, Tuple, Union

import numpy as np
import torch
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MaxAbsScaler
from torch.utils.data import DataLoader

from ..models.model_utils import calc_anisotropy, calc_intrinsic_dimension
from ..models.mTAND.model import MegaNetCE
from .base_trainer import BaseTrainer

logger = logging.getLogger("event_seq")

params_fast = {
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
    "n_jobs": 16,
    "reg_alpha": None,
    "reg_lambda": None,
    "colsample_bytree": None,
    "min_child_samples": None,
    "verbosity": -1,
}
# taken from coles age
params = {
    "n_estimators": 1000,
    "boosting_type": "gbdt",
    # "objective": "binary",
    # "metric": "auc",
    "subsample": 0.75,
    "subsample_freq": 1,
    "learning_rate": 0.02,
    "feature_fraction": 0.75,
    "max_depth": 12,
    "lambda_l1": 1,
    "lambda_l2": 1,
    "min_data_in_leaf": 50,
    "random_state": 42,
    "n_jobs": 16,
    "num_leaves": 50,
    "reg_alpha": None,
    "reg_lambda": None,
    "colsample_bytree": None,
    "min_child_samples": None,
    "verbosity":-1,
}


class SimpleTrainerContrastive(BaseTrainer):
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
        with torch.no_grad():
            loss_dicts = [
                self.model.loss(it.to(self._device), gt.to(self._device))
                for it, gt in zip(model_outputs, ground_truths)
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
        self,
        train_supervised_loader: DataLoader,
        other_loaders: list,
        # cv=False,
    ) -> None:
        """
        Logs test metrics with self.compute_metrics
        """

        logger.info("Test started")
        predict_limit = self._data_conf.get("predict_limit", None)
        train_embeddings, train_gts = self.predict(
            train_supervised_loader, predict_limit
        )
        other_embeddings, other_gts = [], []
        for loader in other_loaders:
            other_embedding, other_gt = (
                self.predict(loader, predict_limit) if len(loader) > 0 else (None, None)
            )
            other_embeddings.append(other_embedding), other_gts.append(other_gt)

        anisotropy = calc_anisotropy(train_embeddings, other_embeddings).item()
        logger.info("Anisotropy: %s", str(anisotropy))

        intrinsic_dimension = calc_intrinsic_dimension(
            train_embeddings, other_embeddings
        )
        logger.info("Intrinsic Dimension: %s", str(intrinsic_dimension))

        (
            train_metric,
            other_metrics,
            train_logist,
            other_logist,
        ) = self.compute_test_metric(
            train_embeddings, train_gts, other_embeddings, other_gts
        )
        logger.info("Train metrics: %s", str(train_metric))
        logger.info("Other metrics: %s", str(other_metrics))
        logger.info("Logist Other metrics: %s", str(other_logist))
        logger.info("Test finished")
        return (
            train_metric,
            other_metrics,
            train_logist,
            other_logist,
            anisotropy,
            intrinsic_dimension,
        )

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

        train_emb_subset = train_embeddings
        train_labels_subset = train_labels

        model, log_model = self.get_model(np.unique(train_labels).shape[0])
        preprocessor = MaxAbsScaler()

        train_emb_subset = preprocessor.fit_transform(train_emb_subset)
        logger.info("Started lgbm")
        model.fit(train_emb_subset, train_labels_subset)
        logger.info("Finished lgbm")
        log_model.fit(train_emb_subset, train_labels_subset)
        logger.info("Finished logistic")
        train_metric = self.get_metric(model, train_emb_subset, train_labels_subset)
        train_logist = self.get_metric(log_model, train_emb_subset, train_labels_subset)

        other_metrics = []
        other_logist = []
        for i, (other_embedding, other_label) in enumerate(
            zip(other_embeddings_new, other_labels)
        ):
            if other_embedding is not None:
                other_embedding_proccesed = preprocessor.transform(other_embedding)
                other_metrics.append(
                    self.get_metric(model, other_embedding_proccesed, other_label)
                )
                other_logist.append(
                    self.get_metric(log_model, other_embedding_proccesed, other_label)
                )
            else:
                other_metrics.append(0)
                other_logist.append(0)
        return (
            train_metric,
            other_metrics,
            train_logist,
            other_logist,
        )


class AucTrainerContrastive(SimpleTrainerContrastive):
    def get_metric(self, model, x, target):
        pred = model.predict_proba(x)
        auc_score = roc_auc_score(target, pred[:, 1])
        return auc_score

    def get_model(self, n_classes=2):
        assert n_classes == 2
        args = params.copy()
        args["objective"] = "binary"
        args["metric"] = "auc"
        return LGBMClassifier(**args), LogisticRegression(solver="saga")


class AccuracyTrainerContrastive(SimpleTrainerContrastive):
    def get_metric(self, model, x, target):
        pred = model.predict(x)
        acc_score = accuracy_score(target, pred)
        return acc_score

    def get_model(self, n_classes=0):
        if n_classes < 15:
            args = params.copy()
            logistic_params = {"solver": "saga"}
        else:
            print("Using fast algos because n_classes too big")
            args = params_fast.copy()
            logistic_params = {"n_jobs":-1, "multi_class": "ovr", "solver": "saga"}
        return LGBMClassifier(**args), LogisticRegression(**logistic_params)

class MSETrainerContrastive(SimpleTrainerContrastive):
    def get_metric(self, model, x, target):
        pred = model.predict(x)
        score = mean_squared_error(target, pred)
        return score

    def get_model(self, n_classes=0):
        args = params.copy()
        args["objective"] = "regression"
        args['verbosity'] = -1
        return LGBMRegressor(**args), LinearRegression()
