#!/usr/bin/env bash

#SBATCH -p serial
#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH --gpus-per-node=1
#SBATCH --constraint=v100-32g --partition=gpuserial-48core

#SBATCH --account=PAS2787
#SBATCH --time=04:00:00




ml miniconda3/23.3.1-py310 cuda/11.8.0


python training_pipnet_OSC.py