#!/bin/sh

#SBATCH -J DeepLabV3_ResNet101_eval_I_cn
#SBATCH -o ./DeepLabV3_ResNet101_eval_I_cn_id%j.txt
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --time=0-05:00:00
#SBATCH --mem=50G

module load miniconda3
##module load cuda/11.6
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate proj_two

## Run training script
CUDA_LAUNCH_BLOCKING=1 python evaluate.py

conda deactivate
