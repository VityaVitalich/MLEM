#!/bin/bash

EPOCHS=70
DEVICE="cuda"
NAME='TC_200_TR_1H_1L_DRP0.1_GRU64_LN'


python train_base_supervised.py --total-epochs=$EPOCHS --device=$DEVICE --run-name=$NAME