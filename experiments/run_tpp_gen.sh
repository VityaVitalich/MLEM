#!/bin/bash

DEVICE="cuda"
DATA_C="../configs/data_configs/pendulum_coord.py"
MODEL_C="../configs/model_configs/gen/pendulum_coord.py"
CONFIGURATION='gen'
CKPT0='/home/event_seq/experiments/pendulum/ckpt/COORD_100k_GEN/seed_0/epoch__0041_-_total_loss__1.57.ckpt'
CKPT1='/home/event_seq/experiments/pendulum/ckpt/COORD_100k_GEN/seed_1/epoch__0066_-_total_loss__1.524.ckpt'
CKPT2='/home/event_seq/experiments/pendulum/ckpt/COORD_100k_GEN/seed_2/epoch__0063_-_total_loss__1.597.ckpt'

python tpp_eval.py \
    --device=$DEVICE \
    --configuration=$CONFIGURATION \
    --data_config=$DATA_C \
    --model_config=$MODEL_C \
    --ckpt_0=$CKPT0 \
    --ckpt_1=$CKPT1 \
    --ckpt_2=$CKPT2 \
