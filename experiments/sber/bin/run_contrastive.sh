#!/bin/bash

EPOCHS=10
DEVICE="cuda:1"
NAME='test'


python train_contrastive.py --total-epochs=$EPOCHS --device=$DEVICE --run-name=$NAME
