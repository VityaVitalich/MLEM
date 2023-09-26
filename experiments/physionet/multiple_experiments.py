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
    default_model_config = model_configs()
    total_epochs = 1
    log_dir = "./logs/testing/"

    gru_hid_options = [32, 64, 128]

    all_results = {}

    for gru_hid in gru_hid_options:
        cur_conf = default_model_config
        cur_conf["classifier_gru_hidden_dim"] = gru_hid

        run_name = "GRU{}".format(gru_hid)

        test_metrics = run_experiment(
            run_name,
            cur_conf.device,
            total_epochs,
            default_data_config,
            cur_conf,
            None,
            log_dir,
        )

    print(all_results)
