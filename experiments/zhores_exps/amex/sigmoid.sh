#!/bin/bash

#SBATCH --job-name=seq_optuna

#SBATCH --partition=ais-gpu

#SBATCH --mail-type=ALL

#SBATCH --mail-user=d.osin@skoltech.ru

#SBATCH --output=outputs/amex_sigmoid.txt

#SBATCH --time=6-00

#SBATCH --mem=100G

#SBATCH --nodes=1

#SBATCH -c 8

#SBATCH --gpus=1

srun singularity exec --bind /gpfs/gpfs0/d.osin/:/home -f --nv event_seq.sif bash -c "
    cd /home/event_seq/experiments;
    nvidia-smi;
    python pipeline_sigmoid.py \
        --run-name='Sigmoid' \
        --data-conf='../configs/data_configs/sigmoid/amex.py' \
        --model-conf='../configs/model_configs/sigmoid/amex.py' \
        --device='cuda:0' \
        --log-dir='./amex/logs/' \
        --total-epochs=30 \
        --console-lvl='info' \
        --FT=0 &
    wait
"