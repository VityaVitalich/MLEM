#!/bin/bash

EPOCHS=1
EPOCHS_RECON=1
EPOCHS_GEN=1
EPOCHS_JOINT=1
DEVICE="cuda"
NAME='TG_test_dist'
DATA_C="../configs/data_configs/alpha.py"
MODEL_C="../configs/model_configs/gen/alpha.py"
LOG_D="./alpha/logs/"
GENVAL=1
GENVAL_EPOCH=1
RECON_VAL=1
RECON_VAL_EPOCH=1
DRAW=1


python pipeline_timegan_distributed.py \
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
