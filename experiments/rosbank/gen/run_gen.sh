#!/bin/bash

EPOCHS=100
DEVICE="cuda"
NAME='cosine'
RECON=true
GEN=true

python train_gen_unsupervised.py --total-epochs=$EPOCHS --device=$DEVICE --run-name=$NAME --recon-val=$RECON --gen-val=$GEN