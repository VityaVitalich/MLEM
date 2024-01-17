#!/bin/bash

EPOCHS=0
DEVICE="cuda"
NAME='COORD_100k_SIGMOID_10s_1r'
DATA_C="../configs/data_configs/pendulum_coord.py"
MODEL_C="../configs/model_configs/sigmoid/pendulum_coord.py"
LOG_D="./pendulum/logs_coord/"
FT=0


python pipeline_sigmoid.py \
    --run-name=$NAME \
    --data-conf=$DATA_C \
    --model-conf=$MODEL_C \
    --device=$DEVICE \
    --log-dir=$LOG_D \
    --total-epochs=$EPOCHS \
    --FT=$FT
