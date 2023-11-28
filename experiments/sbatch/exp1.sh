#!/bin/bash

#SBATCH --job-name=seq_optuna

#SBATCH --partition=ais-gpu

#SBATCH --mail-type=ALL

#SBATCH --mail-user=d.osin@skoltech.ru

#SBATCH --output=outputs/output_1.txt

#SBATCH --time=6-00

#SBATCH --mem=400G

#SBATCH --nodes=1

#SBATCH -c 32

#SBATCH --gpus=4

srun singularity exec --bind /gpfs/gpfs0/d.osin/:/home -f --nv event_seq.sif bash -c '
    cd /home/event_seq/experiments;
    nvidia-smi;
    (sleep 90; sh run_pipe_contrastive.sh cuda:0 contrastive contrasive_loss_27_11_23 age) &
    (sleep 60; sh run_pipe_contrastive.sh cuda:1 contrastive contrasive_loss_27_11_23 age) &
    (sleep 30; sh run_pipe_contrastive.sh cuda:2 contrastive contrasive_loss_27_11_23 age) &
    sh run_pipe_contrastive.sh cuda:3 contrastive contrasive_loss_27_11_23 age &
    wait
'