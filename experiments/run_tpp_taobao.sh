#!/bin/bash

DEVICE="cuda"

############################################################################################################################
CONFIGURATION='contrastive'
DATA_C="../configs/data_configs/contrastive/taobao.py"
MODEL_C="../configs/model_configs/contrastive/age.py"

CKPT0='/home/dev/2023_03/Datasets/event_seq/experiments/taobao/logs/Contrastive/seed_0/ckpt/Contrastive/seed_0/epoch__0100.ckpt'
CKPT1='/home/dev/2023_03/Datasets/event_seq/experiments/taobao/logs/Contrastive/seed_1/ckpt/Contrastive/seed_1/epoch__0100.ckpt'
CKPT2='/home/dev/2023_03/Datasets/event_seq/experiments/taobao/logs/Contrastive/seed_2/ckpt/Contrastive/seed_2/epoch__0100.ckpt'

python tpp_eval.py \
    --device=$DEVICE \
    --configuration=$CONFIGURATION \
    --data_config=$DATA_C \
    --model_config=$MODEL_C \
    --ckpt_0=$CKPT0 \
    --ckpt_1=$CKPT1 \
    --ckpt_2=$CKPT2 \
############################################################################################################################
CONFIGURATION='gen_contrastive'
DATA_C="../configs/data_configs/contrastive/taobao.py"
MODEL_C="../configs/model_configs/GC/age.py"

CKPT0='/home/dev/2023_03/Datasets/event_seq/experiments/taobao/ckpt/GC/seed_0/epoch__0096_-_total_loss__665.1.ckpt'
CKPT1='/home/dev/2023_03/Datasets/event_seq/experiments/taobao/ckpt/GC/seed_1/epoch__0089_-_total_loss__665.4.ckpt'
CKPT2='/home/dev/2023_03/Datasets/event_seq/experiments/taobao/ckpt/GC/seed_2/epoch__0092_-_total_loss__636.3.ckpt'

python tpp_eval.py \
    --device=$DEVICE \
    --configuration=$CONFIGURATION \
    --data_config=$DATA_C \
    --model_config=$MODEL_C \
    --ckpt_0=$CKPT0 \
    --ckpt_1=$CKPT1 \
    --ckpt_2=$CKPT2 \

############################################################################################################################
CONFIGURATION='gen'
DATA_C="../configs/data_configs/taobao.py"
MODEL_C="../configs/model_configs/gen/age.py"

CKPT0='/home/dev/2023_03/Datasets/event_seq/experiments/taobao/ckpt/Gen/seed_0/epoch__0100_-_total_loss__468.2.ckpt'
CKPT1='/home/dev/2023_03/Datasets/event_seq/experiments/taobao/ckpt/Gen/seed_1/epoch__0098_-_total_loss__474.1.ckpt'
CKPT2='/home/dev/2023_03/Datasets/event_seq/experiments/taobao/ckpt/Gen/seed_2/epoch__0098_-_total_loss__463.6.ckpt'

python tpp_eval.py \
    --device=$DEVICE \
    --configuration=$CONFIGURATION \
    --data_config=$DATA_C \
    --model_config=$MODEL_C \
    --ckpt_0=$CKPT0 \
    --ckpt_1=$CKPT1 \
    --ckpt_2=$CKPT2 \

