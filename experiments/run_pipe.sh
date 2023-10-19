#!/bin/bash

EPOCHS=70
DEVICE="cuda:3"
NAME='GRID'
N_RUNS=3
DATASET="age"

python pipeline.py --total-epochs=$EPOCHS --device=$DEVICE --run-name=$NAME --n-runs=$N_RUNS --dataset=$DATASET