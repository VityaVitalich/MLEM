#!/bin/bash

EPOCHS=40
DEVICE='cuda:0'
NAME='CONTRASTIVE_GRU512-32emb'
DATA_C="../configs/data_configs/contrastive/alpha.py"
MODEL_C="../configs/model_configs/contrastive/alpha.py"
LOG_D="./alpha/logs/"
GRID_NAME=''

python pipeline_contrastive.py \
    --run-name=$NAME \
    --data-conf=$DATA_C \
    --model-conf=$MODEL_C \
    --device=$DEVICE \
    --log-dir=$LOG_D \
    --total-epochs=$EPOCHS \
    --grid-name=$GRID_NAME \
    --console-lvl="info" 