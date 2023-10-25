#!/bin/bash

EPOCHS=70
DEVICE="cuda:2"
NAME='optuna_ContrastiveLoss'
N_RUNS=3
DATASET="age"

python contrastive_pipeline.py --total-epochs=$EPOCHS --device=$DEVICE --run-name=$NAME --n-runs=$N_RUNS --dataset=$DATASET