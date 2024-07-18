#!/usr/bin/env bash

#SBATCH -p serial
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=1
#SBATCH --constraint=v100-32g --partition=gpuserial-48core
#SBATCH --account=PAS2787
#SBATCH --time=04:00:00




ml miniconda3/24.1.2-py310 cuda/11.8.0
source activate pytorch

python training_pipnet_OSC.py

# sinteractive -A PAS2787 -N 1 -n 1 -c 8 -g 1 -p gpudebug -t 00:30:00