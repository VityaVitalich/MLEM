#!/bin/bash

#SBATCH --job-name=seq_optuna

#SBATCH --partition=ais-gpu

#SBATCH --mail-type=ALL

#SBATCH --mail-user=d.osin@skoltech.ru

#SBATCH --output=outputs/tao_regr_gen_contrastive.txt

#SBATCH --time=6-00

#SBATCH --mem=100G

#SBATCH --nodes=1

#SBATCH -c 8

#SBATCH --gpus=1

srun singularity exec --bind /gpfs/gpfs0/d.osin/:/home -f --nv event_seq.sif bash -c '
    cd /home/event_seq/experiments;
    nvidia-smi;
    python pipeline_gen.py \
        --run-name="GC" \
        --data-conf="../configs/data_configs/contrastive/taobao_regr.py" \
        --model-conf="../configs/model_configs/GC/age.py" \
        --device="cuda:0" \
        --log-dir="./taobao_regr/logs/" \
        --total-epochs=100 \
        --gen-val=0 \
        --gen-val-epoch=10 \
        --recon-val=0 \
        --recon-val-epoch=10  \
        --draw=0 \
        --FT=0 &
    wait
'