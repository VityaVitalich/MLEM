#!/bin/bash

EPOCHS=100
DEVICE="cuda"
NAME='NEW_GC-1-1_GRU512-TR128-3l-LayerNorm'
DATA_C="../configs/data_configs/contrastive/pendulum.py"
#/home/event_seq/experiments/rosbank/logs/NEW_GC-1-1_GRU512-TR128-3l-LayerNorm
MODEL_C="../configs/model_configs/GC/pendulum.py"
LOG_D="./pendulum/logs/"
GENVAL=0
GENVAL_EPOCH=10
RECON_VAL=0
RECON_VAL_EPOCH=10
DRAW=0
FT=1
#RESUME="/home/event_seq/experiments/alpha/ckpt/GEN_GRU512-TR128-3l-LayerNorm/seed_0/epoch__0037_-_total_loss__1469.ckpt"


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
    --FT=$FT \
  #  --resume=$RESUME
