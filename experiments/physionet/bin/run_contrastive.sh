#!/bin/bash

EPOCHS=30
DEVICE="cuda:2"
NAME='InfoNCE_t_003_all_pos_pairs'


python train_contrastive.py --total-epochs=$EPOCHS --device=$DEVICE --run-name=$NAME
