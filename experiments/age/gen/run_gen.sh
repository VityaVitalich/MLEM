#!/bin/bash

EPOCHS=30
DEVICE="cuda"
NAME='10_cosine_mean'
RECON=true
RECON_EPOCHS=15
GEN=true
GEN_EPOCHS=15

python train_gen_unsupervised.py --total-epochs=$EPOCHS --device=$DEVICE --run-name=$NAME \
 --recon-val=$RECON --recon-val-epoch=$RECON_EPOCHS --gen-val=$GEN --gen-val-epoch=$GEN_EPOCHS