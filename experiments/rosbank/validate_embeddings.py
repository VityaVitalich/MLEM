from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.metrics import roc_auc_score
import sys

sys.path.append("../../")
from src.create_embeddings import create_embeddings
from configs.data_configs.rosbank_inference import data_configs
from configs.model_configs.mTAN.rosbank import model_configs

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
    create_embeddings(conf, model_conf)

    train_embeds = pd.read_csv(conf.train_embed_path, index_col=0)
    test_embeds = pd.read_csv(conf.test_embed_path, index_col=0)

    train_y = pd.read_parquet(conf.train_path)[conf.features.target_col]
    test_y = pd.read_parquet(conf.test_path)[conf.features.target_col].astype(int)

    train = train_embeds.join(train_y)
    train = train.dropna()
    train[conf.features.target_col] = train[conf.features.target_col].astype(int)

    test = test_embeds.join(test_y)

    model = LGBMClassifier(**params)

    model.fit(
        train.drop(columns=[conf.features.target_col]), train[conf.features.target_col]
    )
    y_pred = model.predict(test.drop(columns=[conf.features.target_col]))

    auc_score = roc_auc_score(test_y, y_pred)

    print(auc_score)
