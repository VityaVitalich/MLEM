#!/bin/bash

EPOCHS=60
DEVICE="cuda"
NAME='TC50_TR_1H_1L_drp0.3_GRU64_bs512'


python train_base_supervised.py --total-epochs=$EPOCHS --device=$DEVICE --run-name=$NAME