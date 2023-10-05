import logging
from typing import Any, Dict, List, Literal, Union, Tuple

import numpy as np
import torch

from ..models.mTAND.model import MegaNetCE
from .base_trainer import BaseTrainer
from sklearn.metrics import roc_auc_score, accuracy_score

from lightgbm import LGBMClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

params = {
    "n_estimators": 500,
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
