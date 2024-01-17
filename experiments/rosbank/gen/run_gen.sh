#!/bin/bash

EPOCHS=60
DEVICE="cuda"
NAME='cls_token'
RECON=true
RECON_EPOCHS=20
GEN=true
GEN_EPOCHS=20

python train_gen_unsupervised.py --total-epochs=$EPOCHS --device=$DEVICE --run-name=$NAME \
 --recon-val=$RECON --recon-val-epoch=$RECON_EPOCHS --gen-val=$GEN --gen-val-epoch=$GEN_EPOCHS
