#!/bin/bash

EPOCHS=10
DEVICE="cuda"
NAME='testing_c'


python train_contrastive.py --total-epochs=$EPOCHS --device=$DEVICE --run-name=$NAME