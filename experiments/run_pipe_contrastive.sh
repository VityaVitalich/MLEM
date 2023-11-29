#!/bin/bash

EPOCHS=50
DEVICE="cuda"
NAME='sample-slices-128-50ep'
DATA_C="../configs/data_configs/contrastive/rosbank.py"
MODEL_C="../configs/model_configs/contrastive/rosbank.py"
LOG_D="./rosbank/logs/"

python pipeline_contrastive.py \
    --run-name=$NAME \
    --data-conf=$DATA_C \
    --model-conf=$MODEL_C \
    --device=$DEVICE \
    --log-dir=$LOG_D \
    --total-epochs=$EPOCHS \
