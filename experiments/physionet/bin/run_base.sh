#!/bin/bash

EPOCHS=5
DEVICE="cuda"
NAME='MTS4,16,32,64_GRU128_LN'


python train_base_supervised.py --total-epochs=$EPOCHS --device=$DEVICE --run-name=$NAME