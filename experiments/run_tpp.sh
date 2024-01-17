#!/bin/bash

DEVICE="cuda"
DATA_C="../configs/data_configs/contrastive/alpha.py"
MODEL_C="../configs/model_configs/contrastive/alpha.py"
CONFIGURATION='contrastive'
CKPT0='/home/event_seq/experiments/alpha/logs/CONTRASTIVE_GRU512-32emb/seed_0/ckpt/CONTRASTIVE_GRU512-32emb/seed_0/epoch__0040.ckpt'
CKPT1='/home/event_seq/experiments/alpha/logs/CONTRASTIVE_GRU512-32emb/seed_1/ckpt/CONTRASTIVE_GRU512-32emb/seed_1/epoch__0040.ckpt'
CKPT2='/home/event_seq/experiments/alpha/logs/CONTRASTIVE_GRU512-32emb/seed_2/ckpt/CONTRASTIVE_GRU512-32emb/seed_2/epoch__0040.ckpt'

python tpp_eval.py \
    --device=$DEVICE \
    --configuration=$CONFIGURATION \
    --data_config=$DATA_C \
    --model_config=$MODEL_C \
    --ckpt_0=$CKPT0 \
    --ckpt_1=$CKPT1 \
    --ckpt_2=$CKPT2 \
