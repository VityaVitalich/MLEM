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
    total_epochs = 70
    log_dir = "./logs/MTS_entropy/"

    entropy_weight_options = [0.01, 0.1, 0.2, 0.3, 0.4]
    default_model_config["num_time_blocks"] = [
        2,
        3,
        4,
        5,
        6,
        8,
        10,
        12,
        15,
        20,
        24,
        30,
        40,
        60,
        120,
    ]

    all_results = {}

    for ew in entropy_weight_options:
        cur_conf = default_model_config
        cur_conf["entropy_weight"] = ew

        run_name = "MTS_GRU64_Entropy{}".format(ew)

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
