#!/bin/bash

EPOCHS=1
EPOCHS_RECON=20
EPOCHS_GEN=10
EPOCHS_JOINT=25
DEVICE="cuda"
NAME='TG_128_1l_detach_latens_longer'
DATA_C="../configs/data_configs/rosbank.py"
MODEL_C="../configs/model_configs/gen/rosbank.py"
LOG_D="./rosbank/logs/"
GENVAL=True
GENVAL_EPOCH=5
RECON_VAL=True
RECON_VAL_EPOCH=5

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
    --recon-val-epoch=$RECON_VAL_EPOCH 
