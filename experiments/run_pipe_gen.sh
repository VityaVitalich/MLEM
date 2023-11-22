#!/bin/bash

EPOCHS=70
DEVICE="cuda"
NAME='seq2seq_32-32-pred-cur_step'
DATA_C="../configs/data_configs/rosbank.py"
MODEL_C="../configs/model_configs/gen/rosbank.py"
LOG_D="./rosbank/logs/"
GENVAL=0
GENVAL_EPOCH=20
RECON_VAL=1
RECON_VAL_EPOCH=20
DRAW=0

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
    --draw=$DRAW
