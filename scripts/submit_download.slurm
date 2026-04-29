#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu
#SBATCH --account=begd-dtai-gh
#SBATCH --job-name=GWDownload
#SBATCH --time=02:00:00
#SBATCH --constraint="projects"

module load python
module load python/miniforge3_pytorch/2.5.0
module load cray-hdf5/1.14.3.7

module list

cd "$SLURM_SUBMIT_DIR"

srun $(which python) downloader.py \
    --gps_start 1248652818 \
    --gps_end 1249862418 \
    --sample_rate 4096 \
    --window_len_s 4096 \
    --mode noise \
    --require_full_window \
    --n_segments 5 \
    --output_dir "/projects/begd/victoria/AttenGW/noise/jun2019/" \
    --plot_timeline \
    --plot_timeseries \
    --plot_psd \
    --glitch_sigma 3.0 \
    --glitch_max_frac 0.005 \
    --max_std_ratio 5.0
#add this flag to produce pre-whitened noise segments
   #--whiten \
