#!/bin/bash

EPOCHS=100
DEVICE="cuda:0"
NAME='Sigmoid'
DATA_C="../configs/data_configs/sigmoid/age.py"
MODEL_C="../configs/model_configs/sigmoid/age.py"
LOG_D="./age/logs/"
FT=1


python pipeline_sigmoid.py \
    --run-name=$NAME \
    --data-conf=$DATA_C \
    --model-conf=$MODEL_C \
    --device=$DEVICE \
    --log-dir=$LOG_D \
    --total-epochs=$EPOCHS \
    --FT=$FT
