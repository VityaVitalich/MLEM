#!/bin/bash

EPOCHS=250
DEVICE="cuda"
NAME='SIGMOID_10s_1r_512BATCH_LONG'
DATA_C="../configs/data_configs/rosbank.py"
MODEL_C="../configs/model_configs/sigmoid/rosbank.py"
LOG_D="./rosbank/logs/"
FT=0


python pipeline_sigmoid.py \
    --run-name=$NAME \
    --data-conf=$DATA_C \
    --model-conf=$MODEL_C \
    --device=$DEVICE \
    --log-dir=$LOG_D \
    --total-epochs=$EPOCHS \
    --FT=$FT
