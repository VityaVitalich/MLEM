#!/bin/bash

EPOCHS=30
DEVICE="cuda"
NAME='GRU512-noFM'
DATA_C="../configs/data_configs/alpha.py"
MODEL_C="../configs/model_configs/supervised/alpha.py"
LOG_D="./alpha/logs/"

python pipeline_supervised.py \
    --run-name=$NAME \
    --data-conf=$DATA_C \
    --model-conf=$MODEL_C \
    --device=$DEVICE \
    --log-dir=$LOG_D \
    --total-epochs=$EPOCHS \
