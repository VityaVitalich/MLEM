#!/bin/bash

EPOCHS=0
DEVICE="cuda"
NAME='contrastive_emb'
DATA_C="../configs/data_configs/contrastive/rosbank.py"
MODEL_C="../configs/model_configs/contrastive/rosbank.py"
LOG_D="./rosbank/logs/"
GENVAL=0
GENVAL_EPOCH=10
RECON_VAL=0
RECON_VAL_EPOCH=10
DRAW=0
FT=0
RESUME="/home/event_seq/experiments/rosbank/logs/NEW_CONTRASTIVE_GRU512-32emb/seed_0/ckpt/NEW_CONTRASTIVE_GRU512-32emb/seed_0/epoch__0100.ckpt"


# python extract_embeddings.py \
#     --run-name=$NAME \
#     --data-conf=$DATA_C \
#     --model-conf=$MODEL_C \
#     --device=$DEVICE \
#     --log-dir=$LOG_D \
#     --total-epochs=$EPOCHS \
#     --gen-val=$GENVAL \
#     --gen-val-epoch=$GENVAL_EPOCH \
#     --recon-val=$RECON_VAL \
#     --recon-val-epoch=$RECON_VAL_EPOCH  \
#     --draw=$DRAW \
#     --FT=$FT \
#     --console-lvl="info" \
#     --resume=$RESUME

python extract_embeddings_contrastive.py \
    --run-name=$NAME \
    --data-conf=$DATA_C \
    --model-conf=$MODEL_C \
    --device=$DEVICE \
    --log-dir=$LOG_D \
    --total-epochs=$EPOCHS \
    --FT=$FT \
    --console-lvl="info" \
    --resume=$RESUME