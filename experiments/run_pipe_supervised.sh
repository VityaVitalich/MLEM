#!/bin/bash

EPOCHS=1
DEVICE="cuda:1"
NAME='debug_p2'
DATA_C="../configs/data_configs/age.py"
MODEL_C="../configs/model_configs/supervised/age.py"
LOG_D="./age/logs/"

python pipeline_supervised.py \
    --run-name=$NAME \
    --data-conf=$DATA_C \
    --model-conf=$MODEL_C \
    --device=$DEVICE \
    --log-dir=$LOG_D \
    --total-epochs=$EPOCHS \
