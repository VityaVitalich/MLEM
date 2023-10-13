#!/bin/bash

EPOCHS=1
DEVICE="cuda"
NAME='test_CS_GRU64_LN'


python train_contrastive.py --total-epochs=$EPOCHS --device=$DEVICE --run-name=$NAME