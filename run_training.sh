#!/bin/sh

#SBATCH -J DeepLabV3_ResNet101_train_ILDE_dice
#SBATCH -o ./DeepLabV3_ResNet101_train_ILDE_dice_id%j.txt
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --time=0-23:59:59
#SBATCH --mem=56G

module load miniconda3
##module load cuda/11.6
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate proj_two

## Run training script
CUDA_LAUNCH_BLOCKING=1 python train.py

conda deactivate
