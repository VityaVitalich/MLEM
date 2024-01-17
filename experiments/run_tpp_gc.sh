#!/bin/bash

DEVICE="cuda"
DATA_C="../configs/data_configs/contrastive/pendulum_coord.py"
MODEL_C="../configs/model_configs/GC/pendulum_coord.py"
CONFIGURATION='gen_contrastive'
CKPT0='/home/event_seq/experiments/pendulum/ckpt/COORD_100K_GC/seed_0/epoch__0070_-_total_loss__1.316.ckpt'
CKPT1='/home/event_seq/experiments/pendulum/ckpt/COORD_100K_GC/seed_1/epoch__0024_-_total_loss__1.216.ckpt'
CKPT2='/home/event_seq/experiments/pendulum/ckpt/COORD_100K_GC/seed_2/epoch__0054_-_total_loss__1.215.ckpt'

python tpp_eval.py \
    --device=$DEVICE \
    --configuration=$CONFIGURATION \
    --data_config=$DATA_C \
    --model_config=$MODEL_C \
    --ckpt_0=$CKPT0 \
    --ckpt_1=$CKPT1 \
    --ckpt_2=$CKPT2 \
