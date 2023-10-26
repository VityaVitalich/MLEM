#!/bin/bash

EPOCHS=1
DEVICE="cuda"
NAME='test2'
N_RUNS=1
DATASET="rosbank"

python pipeline.py --total-epochs=$EPOCHS --device=$DEVICE --run-name=$NAME --n-runs=$N_RUNS --dataset=$DATASET