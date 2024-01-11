#!/bin/bash

EPOCHS=10
DEVICE="cuda"
NAME='COORD_100K_Sigmoid_FT'
DATA_C="../configs/data_configs/pendulum_coord.py"
MODEL_C="../configs/model_configs/supervised/pendulum_coord.py"
LOG_D="./pendulum/logs_coord/"
# Contrastive
# CKPT0='/home/event_seq/experiments/pendulum/logs_coord/COORD_100K_CONTRASTIVE/seed_0/ckpt/COORD_100K_CONTRASTIVE/seed_0/epoch__0070.ckpt'
# CKPT1='/home/event_seq/experiments/pendulum/logs_coord/COORD_100K_CONTRASTIVE/seed_1/ckpt/COORD_100K_CONTRASTIVE/seed_1/epoch__0070.ckpt'
# CKPT2='/home/event_seq/experiments/pendulum/logs_coord/COORD_100K_CONTRASTIVE/seed_2/ckpt/COORD_100K_CONTRASTIVE/seed_2/epoch__0070.ckpt'
# GEN
# CKPT0='/home/event_seq/experiments/pendulum/ckpt/COORD_100k_GEN/seed_0/epoch__0041_-_total_loss__1.57.ckpt'
# CKPT1='/home/event_seq/experiments/pendulum/ckpt/COORD_100k_GEN/seed_1/epoch__0066_-_total_loss__1.524.ckpt'
# CKPT2='/home/event_seq/experiments/pendulum/ckpt/COORD_100k_GEN/seed_2/epoch__0063_-_total_loss__1.597.ckpt'
#GC
# CKPT0='/home/event_seq/experiments/pendulum/ckpt/COORD_100K_GC/seed_0/epoch__0070_-_total_loss__1.316.ckpt'
# CKPT1='/home/event_seq/experiments/pendulum/ckpt/COORD_100K_GC/seed_1/epoch__0024_-_total_loss__1.216.ckpt'
# CKPT2='/home/event_seq/experiments/pendulum/ckpt/COORD_100K_GC/seed_2/epoch__0054_-_total_loss__1.215.ckpt'

# Sigmoid
# CKPT0='/home/event_seq/experiments/pendulum/ckpt/COORD_100k_SIGMOID_10s_1r/seed_0/epoch__0030_-_total_loss__2.589.ckpt'
# CKPT1='/home/event_seq/experiments/pendulum/ckpt/COORD_100k_SIGMOID_10s_1r/seed_1/epoch__0036_-_total_loss__2.407.ckpt'
# CKPT2='/home/event_seq/experiments/pendulum/ckpt/COORD_100k_SIGMOID_10s_1r/seed_2/epoch__0058_-_total_loss__2.426.ckpt'

python pipeline_supervised.py \
    --run-name=$NAME \
    --data-conf=$DATA_C \
    --model-conf=$MODEL_C \
    --device=$DEVICE \
    --log-dir=$LOG_D \
    --total-epochs=$EPOCHS \
    # --resume_list=$CKPT0 \
    # --resume_list=$CKPT1 \
    # --resume_list=$CKPT2 \
