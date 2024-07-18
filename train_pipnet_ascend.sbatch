#!/usr/bin/env bash

#SBATCH -p serial
#SBATCH -N 1
#SBATCH --account=PAS2787
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32000mb
#SBATCH --time=06:00:00
#SBATCH --output=train_pip3_ascend_0.out








ml miniconda3/23.3.1-py310 cuda/11.8.0
source activate pytorch
# source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh




python training_pipnet_OSC.py