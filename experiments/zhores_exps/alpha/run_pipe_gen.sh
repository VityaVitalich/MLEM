#!/bin/bash

EPOCHS=40
DEVICE="cuda:0"
NAME='GEN_GRU512-TR128-3l-2h'
DATA_C="../configs/data_configs/alpha.py"
MODEL_C="../configs/model_configs/gen/alpha.py"
LOG_D="./alpha/logs/"
GENVAL=0
GENVAL_EPOCH=10
RECON_VAL=0
RECON_VAL_EPOCH=10
DRAW=0
FT=1


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
