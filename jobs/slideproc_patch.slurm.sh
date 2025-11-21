#!/bin/bash
#SBATCH --job-name=slideproc-patch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slideproc-patch-%j.out
#SBATCH --error=logs/slideproc-patch-%j.err

set -euo pipefail

##############################
# User variables
##############################
WSI_ROOT=""
OUTPUT_ROOT=""
SAM_CHECKPOINT="sam2_bs2_inp1024_lr5e-4.pth"
TARGET_MAG=20
PATCH_SIZE=256
SEG_BATCH=32

# Activate your environment (edit as needed)
# source ~/.bashrc
# conda activate slideprocessor

##############################
# Derived settings
##############################
PATCH_WORKERS="${SLURM_CPUS_PER_TASK:-8}"
MAX_OPEN_SLIDES=200

mkdir -p logs

slideproc process "${WSI_ROOT}" \
    --checkpoint "${SAM_CHECKPOINT}" \
    --patch-size "${PATCH_SIZE}" \
    --target-mag "${TARGET_MAG}" \
    --output "${OUTPUT_ROOT}" \
    --tissue-thresh 0 \
    --fast-mode \
    --recursive \
    --seg-batch-size "${SEG_BATCH}" \
    --patch-workers "${PATCH_WORKERS}" \
    --max-open-slides "${MAX_OPEN_SLIDES}"

# To enable per-patch content filtering, append: --no-fast-mode
