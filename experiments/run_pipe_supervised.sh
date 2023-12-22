#!/bin/bash

EPOCHS=70
DEVICE="cuda"
NAME='GRU512-supervised'
DATA_C="../configs/data_configs/pendulum.py"
MODEL_C="../configs/model_configs/supervised/pendulum.py"
LOG_D="./pendulum/logs/"

python pipeline_supervised.py \
    --run-name=$NAME \
    --data-conf=$DATA_C \
    --model-conf=$MODEL_C \
    --device=$DEVICE \
    --log-dir=$LOG_D \
    --total-epochs=$EPOCHS \
