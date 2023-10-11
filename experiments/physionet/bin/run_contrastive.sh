#!/bin/bash

EPOCHS=100
DEVICE="cuda"
NAME='test_CS_TR_1H_1L_DRP0.2_GRU512'


python train_contrastive.py --total-epochs=$EPOCHS --device=$DEVICE --run-name=$NAME