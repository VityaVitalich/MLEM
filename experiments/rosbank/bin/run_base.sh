#!/bin/bash

EPOCHS=40
DEVICE="cuda"
NAME='MTS_GRU128_LN'


python train_base_supervised.py --total-epochs=$EPOCHS --device=$DEVICE --run-name=$NAME