#!/bin/bash

#SBATCH --job-name=seq_optuna

#SBATCH --partition=ais-gpu

#SBATCH --mail-type=ALL

#SBATCH --mail-user=d.osin@skoltech.ru

#SBATCH --output=outputs/output_6.txt
#SBATCH --error=/dev/null

#SBATCH --time=6-00

#SBATCH --mem=100G

#SBATCH --nodes=1

#SBATCH -c 8

#SBATCH --gpus=1

srun singularity exec --bind /gpfs/gpfs0/d.osin/:/home -f --nv event_seq.sif bash -c '
    cd /home/event_seq/experiments;
    nvidia-smi;
    (sh run_pipe_contrastive.sh cuda:0 DecoupledInfoNCELoss decoupledinfoloss_27_11_23 age) &
    wait
'