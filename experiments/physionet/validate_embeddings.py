from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.metrics import roc_auc_score
import sys
import torch

sys.path.append("../../")
from src.create_embeddings import create_embeddings
from configs.data_configs.physionet_contrastive import data_configs
from configs.model_configs.mTAN.physionet import model_configs
from sklearn.preprocessing import MaxAbsScaler

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


if __name__ == "__main__":
    conf = data_configs()
    model_conf = model_configs()
    train_embeddings, train_gts, test_embeddings, test_gts = create_embeddings(
        conf, model_conf
    )

    train_labels = torch.cat([gt[1].cpu() for gt in train_gts]).numpy()
    train_embeddings = torch.cat(train_embeddings).cpu().numpy()

    test_labels = torch.cat([gt[1].cpu() for gt in test_gts]).numpy()
    test_embeddings = torch.cat(test_embeddings).cpu().numpy()

    model = LGBMClassifier(**params)
    preprocessor = MaxAbsScaler()

    train_embeddings = preprocessor.fit_transform(train_embeddings)
    test_embeddings = preprocessor.transform(test_embeddings)

    model.fit(train_embeddings, train_labels)
    y_pred = model.predict_proba(test_embeddings)

    auc_score = roc_auc_score(test_labels, y_pred[:, 1])

    print(auc_score)
