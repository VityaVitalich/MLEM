#!/bin/bash

EPOCHS=40
DEVICE="cuda:0"
NAME='GC-NO-CLIP'
DATA_C="../configs/data_configs/contrastive/alpha.py"
MODEL_C="../configs/model_configs/GC/alpha.py"
LOG_D="./alpha/logs/"
GENVAL=0
GENVAL_EPOCH=10
RECON_VAL=0
RECON_VAL_EPOCH=10
DRAW=0
FT=0


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
    --recon-val-epoch=$RECON_VAL_EPOCH  \
    --draw=$DRAW \
    --FT=$FT
