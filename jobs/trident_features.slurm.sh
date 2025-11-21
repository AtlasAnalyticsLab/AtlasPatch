#!/bin/bash
#SBATCH --job-name=trident-feat
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/trident-feat-%j.out
#SBATCH --error=logs/trident-feat-%j.err

set -euo pipefail

##############################
# User variables
##############################
WSI_ROOT=""
PATCH_OUTPUT=""
PATCH_ENCODER="uni_v1"
BATCH_SIZE=64
PATCH_SIZE=256
MAG=20

# Activate your TRIDENT environment (edit as needed)
# source ~/.bashrc
# conda activate trident
# export PYTHONPATH=/path/to/TRIDENT:${PYTHONPATH}

##############################
# Command
##############################
mkdir -p logs

python run_batch_of_slides.py \
    --task feat \
    --wsi_dir "${WSI_ROOT}" \
    --job_dir "${PATCH_OUTPUT}" \
    --batch_size "${BATCH_SIZE}" \
    --patch_encoder "${PATCH_ENCODER}" \
    --mag "${MAG}" \
    --patch_size "${PATCH_SIZE}"
