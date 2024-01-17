#!/bin/bash

EPOCHS=100
DEVICE="cuda"
NAME='coles-like'
DATA_C="../configs/data_configs/contrastive/age.py"
MODEL_C="../configs/model_configs/contrastive/age.py"
LOG_D="./age/logs/"

python pipeline_contrastive.py \
    --run-name=$NAME \
    --data-conf=$DATA_C \
    --model-conf=$MODEL_C \
    --device=$DEVICE \
    --log-dir=$LOG_D \
    --total-epochs=$EPOCHS \