#!/bin/bash

DEVICE="cuda"
DATA_C="../configs/data_configs/contrastive/pendulum_coord.py"
MODEL_C="../configs/model_configs/contrastive/pendulum_coord.py"
CONFIGURATION='contrastive'
CKPT0='/home/event_seq/experiments/pendulum/logs_coord/COORD_100K_CONTRASTIVE/seed_0/ckpt/COORD_100K_CONTRASTIVE/seed_0/epoch__0070.ckpt'
CKPT1='/home/event_seq/experiments/pendulum/logs_coord/COORD_100K_CONTRASTIVE/seed_1/ckpt/COORD_100K_CONTRASTIVE/seed_1/epoch__0070.ckpt'
CKPT2='/home/event_seq/experiments/pendulum/logs_coord/COORD_100K_CONTRASTIVE/seed_2/ckpt/COORD_100K_CONTRASTIVE/seed_2/epoch__0070.ckpt'

python tpp_eval.py \
    --device=$DEVICE \
    --configuration=$CONFIGURATION \
    --data_config=$DATA_C \
    --model_config=$MODEL_C \
    --ckpt_0=$CKPT0 \
    --ckpt_1=$CKPT1 \
    --ckpt_2=$CKPT2 \
