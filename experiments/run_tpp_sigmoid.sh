#!/bin/bash

DEVICE="cuda"
DATA_C="../configs/data_configs/pendulum_coord.py"
MODEL_C="../configs/model_configs/sigmoid/pendulum_coord.py"
CONFIGURATION='sigmoid'
CKPT0='/home/event_seq/experiments/pendulum/ckpt/COORD_100k_SIGMOID_10s_1r/seed_0/epoch__0030_-_total_loss__2.589.ckpt'
CKPT1='/home/event_seq/experiments/pendulum/ckpt/COORD_100k_SIGMOID_10s_1r/seed_1/epoch__0036_-_total_loss__2.407.ckpt'
CKPT2='/home/event_seq/experiments/pendulum/ckpt/COORD_100k_SIGMOID_10s_1r/seed_2/epoch__0058_-_total_loss__2.426.ckpt'

python tpp_eval.py \
    --device=$DEVICE \
    --configuration=$CONFIGURATION \
    --data_config=$DATA_C \
    --model_config=$MODEL_C \
    --ckpt_0=$CKPT0 \
    --ckpt_1=$CKPT1 \
    --ckpt_2=$CKPT2 \
