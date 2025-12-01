#!/bin/bash
#SBATCH --job-name=atlaspatch-features
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/atlaspatch-features-%j.out
#SBATCH --error=logs/atlaspatch-features-%j.err

set -euo pipefail

##############################
# User variables
##############################
WSI_ROOT=""        # Same input path used for patch extraction (file or directory)
OUTPUT_ROOT=""     # Same output root that already contains patches/*.h5
TARGET_MAG=20
PATCH_SIZE=256
FEATURES="resnet18,resnet50"  # comma/space separated list
FEATURE_DEVICE="cuda"
FEATURE_BATCH=64
FEATURE_WORKERS="${SLURM_CPUS_PER_TASK:-8}"
FEATURE_PRECISION="float32"  # float32|float16|bfloat16
DEVICE="cuda"      # Segmentation device; kept for slides missing coords

# Activate your environment (edit as needed)
# source ~/.bashrc
# conda activate atlaspatch

mkdir -p logs

atlaspatch process "${WSI_ROOT}" \
  --output "${OUTPUT_ROOT}" \
  --patch-size "${PATCH_SIZE}" \
  --target-mag "${TARGET_MAG}" \
  --feature-extractors "${FEATURES}" \
  --feature-device "${FEATURE_DEVICE}" \
  --feature-batch-size "${FEATURE_BATCH}" \
  --feature-num-workers "${FEATURE_WORKERS}" \
  --feature-precision "${FEATURE_PRECISION}" \
  --device "${DEVICE}" \
  --skip-existing \
  --recursive

# With --skip-existing, slides with existing coords are reused and only missing feature sets are embedded.
