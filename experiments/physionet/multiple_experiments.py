from train_base_supervised import run_experiment
import torch
import sys
import os
import ml_collections
import logging
from pathlib import Path


sys.path.append("../../")

from configs.data_configs.physionet import data_configs
from configs.model_configs.mTAN.physionet import model_configs


if __name__ == "__main__":
    default_data_config = data_configs()
    default_model_config = ml_collections.ConfigDict(
        {
            "model_name": "GRUClassifier",
            "features_emb_dim": 8,
            "classifier_gru_hidden_dim": 128,
            "classifier_linear_hidden_dim": 300,
            "encoder": "TransformerEncoder",
            "num_enc_layers": 1,
            "num_heads_enc": 1,
            "pre_gru_norm": "Identity",
            "post_gru_norm": "LayerNorm",
            "encoder_norm": "Identity",
            "conv": {
                "out_channels": 32,
                "kernels": [3, 5, 9],
                "dilations": [3, 5, 9],
                "num_stacks": 3,
                "proj": "Linear",
            },
            "after_enc_dropout": 0.0,
            "activation": "ReLU",
            "num_time_blocks": [4],
            "time_preproc": "MultiTimeSummator",
            "num_ref_points": 128,
            "latent_dim": 2,
            "ref_point_dim": 128,
            "time_emb_dim": 16,
            "num_heads_enc": 2,
            "linear_hidden_dim": 50,
            "num_time_emb": 3,
            "k_iwae": 1,
            "noise_std": 0.01,
            "kl_weight": 0.0,
            "CE_weight": 0,
            "reconstruction_weight": 0,
            "classification_weight": 1,
            "device": "cuda",
            "lr": 3e-3,
            "weight_decay": 1e-3,
        }
    )

    total_epochs = 70
    log_dir = "./logs/MTS_DRP_GRU/"

    tc_options = [[16, 32], [16, 32, 64], [16, 64], [16, 64, 128]]
    gru_hid_options = [32, 64, 128]
    drp_options = [0.0, 0.2, 0.3, 0.4]

    all_results = {}

    for tc in tc_options:
        for drp in drp_options:
            for gru_hid in gru_hid_options:
                cur_conf = default_model_config
                cur_conf["num_time_blocks"] = tc
                cur_conf["after_enc_dropout"] = drp
                cur_conf["classifier_gru_hidden_dim"] = gru_hid

                run_name = "MTS_{}_DRP{}_GRU{}".format(tc, drp, gru_hid)

                test_metrics = run_experiment(
                    run_name,
                    cur_conf.device,
                    total_epochs,
                    default_data_config,
                    cur_conf,
                    None,
                    log_dir,
                )

                all_results[(tuple(tc), drp, gru_hid)] = test_metrics["roc_auc"]

    print(all_results)
