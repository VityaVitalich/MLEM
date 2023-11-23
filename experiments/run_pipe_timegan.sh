#!/bin/bash

EPOCHS=1
EPOCHS_RECON=50
EPOCHS_GEN=50
EPOCHS_JOINT=50
DEVICE="cuda"
NAME='TG_delta-normal-loss'
DATA_C="../configs/data_configs/rosbank.py"
MODEL_C="../configs/model_configs/gen/rosbank.py"
LOG_D="./rosbank/logs/"
GENVAL=1
GENVAL_EPOCH=20
RECON_VAL=1
RECON_VAL_EPOCH=20
DRAW=1


python pipeline_timegan.py \
    --run-name=$NAME \
    --data-conf=$DATA_C \
    --model-conf=$MODEL_C \
    --device=$DEVICE \
    --log-dir=$LOG_D \
    --total-epochs=$EPOCHS \
    --total-epochs-recon=$EPOCHS_RECON \
    --total-epochs-gen=$EPOCHS_GEN \
    --total-epochs-joint=$EPOCHS_JOINT \
    --gen-val=$GENVAL \
    --gen-val-epoch=$GENVAL_EPOCH \
    --recon-val=$RECON_VAL \
    --recon-val-epoch=$RECON_VAL_EPOCH \
    --draw=$DRAW
