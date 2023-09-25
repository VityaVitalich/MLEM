#!/bin/bash

EPOCHS=5
DEVICE="cuda"
NAME='testing'


python train_base_supervised.py --total-epochs=$EPOCHS --device=$DEVICE --run-name=$NAME