#!/bin/bash

EPOCHS=1
DEVICE="cuda"
NAME='test-ft'
DATA_C="../configs/data_configs/rosbank.py"
MODEL_C="../configs/model_configs/gen/rosbank.py"
LOG_D="./rosbank/test_logs/"
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
