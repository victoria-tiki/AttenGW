#!/bin/bash
#SBATCH --job-name=GWDownload
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# Submit this script from the repository root:
#   sbatch scripts/submit_download.slurm

# ===== EDIT THESE =====
NOISE_DIR="/path/to/noise_dir"

GPS_START=1248652818
GPS_END=1249862418
WINDOW_LEN_S=4096
N_SEGMENTS=5
MODE="noise"   # options: noise, signal, both

# ===== SHARED PHYSICAL/DATA SETTINGS =====
# These should match the training/dataloader settings.
SAMPLE_RATE=4096
BAND_LOW=25.0
BAND_HIGH=450.0
BANDPASS_ORDER=4
PSD_SEGLEN_S=4.0
TARGET_PLOT_FS=1024.0

# ===== OPTIONAL CLEANING / QC SETTINGS =====
GLITCH_SIGMA=3.0
GLITCH_MAX_FRAC=0.005
MAX_STD_RATIO=5.0

srun python -u downloader.py \
  --gps_start "${GPS_START}" \
  --gps_end "${GPS_END}" \
  --sample_rate "${SAMPLE_RATE}" \
  --window_len_s "${WINDOW_LEN_S}" \
  --mode "${MODE}" \
  --require_full_window \
  --n_segments "${N_SEGMENTS}" \
  --output_dir "${NOISE_DIR}" \
  --band_low "${BAND_LOW}" \
  --band_high "${BAND_HIGH}" \
  --bandpass_order "${BANDPASS_ORDER}" \
  --psd_seglen_s "${PSD_SEGLEN_S}" \
  --target_plot_fs "${TARGET_PLOT_FS}" \
  --plot_timeline \
  --plot_timeseries \
  --plot_psd \
  --glitch_sigma "${GLITCH_SIGMA}" \
  --glitch_max_frac "${GLITCH_MAX_FRAC}" \
  --max_std_ratio "${MAX_STD_RATIO}"
# If you want to save already-whitened noise files, add:
#   --whiten
# Training should then use:
#   --noise_is_whitened
