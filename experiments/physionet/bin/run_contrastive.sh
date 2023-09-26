#!/bin/bash

EPOCHS=50
DEVICE="cuda"
NAME='CS_GRU128_LN'


python train_contrastive.py --total-epochs=$EPOCHS --device=$DEVICE --run-name=$NAME