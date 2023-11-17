#!/bin/bash

EPOCHS=1
DEVICE="cuda"
NAME='debug_time_emb4_fm'
DATA_C="../configs/data_configs/rosbank.py"
MODEL_C="../configs/model_configs/supervised/rosbank.py"
LOG_D="./rosbank/logs/"

python pipeline_supervised.py \
    --run-name=$NAME \
    --data-conf=$DATA_C \
    --model-conf=$MODEL_C \
    --device=$DEVICE \
    --log-dir=$LOG_D \
    --total-epochs=$EPOCHS \
