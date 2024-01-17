#!/bin/bash

EPOCHS=40
DEVICE="cuda:0"
NAME='Sigmoid'
DATA_C="../configs/data_configs/alpha.py"
MODEL_C="../configs/model_configs/sigmoid/alpha.py"
LOG_D="./alpha/logs/"
FT=0


python pipeline_sigmoid.py \
    --run-name=$NAME \
    --data-conf=$DATA_C \
    --model-conf=$MODEL_C \
    --device=$DEVICE \
    --log-dir=$LOG_D \
    --total-epochs=$EPOCHS \
    --FT=$FT
