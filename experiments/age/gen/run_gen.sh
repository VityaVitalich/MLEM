#!/bin/bash

EPOCHS=15
DEVICE="cuda"
NAME='mixer'
GENVAL=true

python train_gen_unsupervised.py --total-epochs=$EPOCHS --device=$DEVICE --run-name=$NAME --gen-val=$GENVAL