#!/bin/bash
#SBATCH --job-name=AttenGW
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# ===== EDIT THESE =====
DATA_DIR="/path/to/train_test_hdf_dir"
NOISE_DIR="/path/to/noise_dir"
CKPT_DIR="/path/to/checkpoints_dir"
TRAIN_FILE="train.hdf"
VAL_FILE="val.hdf"

# ===== OPTIONAL (check train.py for documentation) =====
BATCH_SIZE=32
NUM_WORKERS=4
NUM_NODES=1
LR_INIT=0.001

NOISE_PROB=0.60
P_HIGHER_INIT=0.9
P_HIGHER_FIN=0.25
SEGMENT_LENGTH=4096
EDGE_BUFFER=2048
DIM=1024

srun python -u train.py \
  --batch_size "${BATCH_SIZE}" \
  --data_dir "${DATA_DIR}" \
  --checkpoint_dir "${CKPT_DIR}" \
  --noise_dir "${NOISE_DIR}" \
  --train_file "${TRAIN_FILE}" \
  --val_file "${VAL_FILE}" \
  --num_workers "${NUM_WORKERS}" \
  --num_nodes "${NUM_NODES}" \
  --lr_init "${LR_INIT}" \
  --noise_prob "${NOISE_PROB}" \
  --p_higher_init "${P_HIGHER_INIT}" \
  --p_higher_fin "${P_HIGHER_FIN}" \
  --segment_length "${SEGMENT_LENGTH}" \
  --edge_buffer "${EDGE_BUFFER}" \
  --dim "${DIM}"
# If your noise files are already whitened, add:
# --noise_is_whitened
