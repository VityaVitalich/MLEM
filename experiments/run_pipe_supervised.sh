#!/bin/bash

EPOCHS=15
DEVICE="cuda"
NAME='GRUD_64h_l1_no_numemb'
DATA_C="../configs/data_configs/physionet.py"
MODEL_C="../configs/model_configs/supervised/physionet.py"
LOG_D="./physionet/logs/"

python pipeline_supervised.py \
    --run-name=$NAME \
    --data-conf=$DATA_C \
    --model-conf=$MODEL_C \
    --device=$DEVICE \
    --log-dir=$LOG_D \
    --total-epochs=$EPOCHS \
