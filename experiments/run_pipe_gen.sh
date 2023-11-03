#!/bin/bash

EPOCHS=1
DEVICE="cuda"
NAME='debug_gen'
DATA_C="../configs/data_configs/age.py"
MODEL_C="../configs/model_configs/gen/age.py"
LOG_D="./age/logs/"
GENVAL=True
GENVAL_EPOCH=1
RECON_VAL=True
RECON_VAL_EPOCH=1

python pipeline_gen.py \
    --run-name=$NAME \
    --data-conf=$DATA_C \
    --model-conf=$MODEL_C \
    --device=$DEVICE \
    --log-dir=$LOG_D \
    --total-epochs=$EPOCHS \
    --gen-val=$GENVAL \
    --gen-val-epoch=$GENVAL_EPOCH \
    --recon-val=$RECON_VAL \
    --recon-val-epoch=$RECON_VAL_EPOCH 
