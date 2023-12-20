#!/bin/bash

EPOCHS=1
DEVICE="cuda"
NAME='test'
DATA_C="../configs/data_configs/rosbank.py"
MODEL_C="../configs/model_configs/sigmoid/rosbank.py"
LOG_D="./rosbank/test_logs/"
FT=1


python pipeline_sigmoid.py \
    --run-name=$NAME \
    --data-conf=$DATA_C \
    --model-conf=$MODEL_C \
    --device=$DEVICE \
    --log-dir=$LOG_D \
    --total-epochs=$EPOCHS \
    --FT=$FT
