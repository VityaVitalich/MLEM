#!/bin/bash

EPOCHS=10
DEVICE="cuda"
NAME='CONTRASTIVE_FT2'
DATA_C="../configs/data_configs/taobao_regr.py"
MODEL_C="../configs/model_configs/supervised/taobao_regr.py"
LOG_D="./taobao_regr/logs/"
# Contrastive
CKPT0='/home/event_seq/experiments/taobao_regr/logs/Contrastive/seed_0/ckpt/Contrastive/seed_0/epoch__0100.ckpt'
CKPT1='/home/event_seq/experiments/taobao_regr/logs/Contrastive/seed_1/ckpt/Contrastive/seed_1/epoch__0100.ckpt'
CKPT2='/home/event_seq/experiments/taobao_regr/logs/Contrastive/seed_2/ckpt/Contrastive/seed_2/epoch__0100.ckpt'
# GEN
# CKPT0='/home/event_seq/experiments/taobao_regr/ckpt/Gen/seed_0/epoch__0100_-_total_loss__468.2.ckpt'
# CKPT1='/home/event_seq/experiments/taobao_regr/ckpt/Gen/seed_1/epoch__0098_-_total_loss__474.1.ckpt'
# CKPT2='/home/event_seq/experiments/taobao_regr/ckpt/Gen/seed_2/epoch__0098_-_total_loss__463.6.ckpt'
#GC
# CKPT0='/home/event_seq/experiments/taobao_regr/ckpt/GC/seed_0/epoch__0096_-_total_loss__665.1.ckpt'
# CKPT1='/home/event_seq/experiments/taobao_regr/ckpt/GC/seed_1/epoch__0089_-_total_loss__665.4.ckpt'
# CKPT2='/home/event_seq/experiments/taobao_regr/ckpt/GC/seed_2/epoch__0092_-_total_loss__636.3.ckpt'

# Sigmoid
# CKPT0='/home/event_seq/experiments/taobao_regr/ckpt/Sigmoid/seed_0/epoch__0098_-_total_loss__485.3.ckpt'
# CKPT1='/home/event_seq/experiments/taobao_regr/ckpt/Sigmoid/seed_1/epoch__0100_-_total_loss__488.4.ckpt'
# CKPT2='/home/event_seq/experiments/taobao_regr/ckpt/Sigmoid/seed_2/epoch__0098_-_total_loss__471.ckpt'

python pipeline_supervised.py \
    --run-name=$NAME \
    --data-conf=$DATA_C \
    --model-conf=$MODEL_C \
    --device=$DEVICE \
    --log-dir=$LOG_D \
    --total-epochs=$EPOCHS \
    --resume_list=$CKPT0 \
    --resume_list=$CKPT1 \
    --resume_list=$CKPT2 \
