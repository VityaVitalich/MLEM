#!/bin/bash

# EPOCHS=100
DEVICE=${1:-'cuda:0'}
NAME=${2:-'contrastive'}
GRID_NAME=${3:-'contrasive_loss_27_11_23'}
DATASET=${4:-'age'}

case $DATASET in
    "age")
        EPOCHS=100
        ;;
    "alpha")
        EPOCHS=5
        ;;
    *)
        EPOCHS=10
        ;;
esac

# DATA_C="../configs/data_configs/contrastive/$DATASET.py"
# MODEL_C="../configs/model_configs/contrastive/$DATASET.py"
# LOG_D="./$DATASET/logs/"
EPOCHS=100
DEVICE='cuda'
NAME='NEW_CONTRASTIVE-GRU512-4emb'
DATA_C="../configs/data_configs/contrastive/pendulum.py"
MODEL_C="../configs/model_configs/contrastive/pendulum.py"
LOG_D="./pendulum/logs/"
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