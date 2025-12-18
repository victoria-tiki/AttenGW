#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16              # was 16
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfxk-delta-gpu
#SBATCH --job-name=wavenet
#SBATCH --time=20:00:00
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=4
#SBATCH --mem=120G                     # <<< NEW: adjust if too high for the queue


module load pytorch-conda
#module load python/anaconda3
module list  


total_gpus=$(($SLURM_JOB_NUM_NODES * $SLURM_GPUS_PER_NODE))
lr_init_scaled=$(echo "0.001 * $total_gpus" | bc -l)                   #$total_gpus gives the number of gpus in case lr scaling is desired

srun $(which python) train_torch.py --batch_size 32 \
                                    --data_dir "/projects/bbvf/victoria/WaveNet_data/combined_spin/" \
                                    --checkpoint_dir "/projects/begd/victoria/Wavenet_torch-main/checkpoints/" \
                                    --noise_dir "/projects/begd/victoria/Wavenet_torch-main/noise/jun2019/" \
                                    --num_workers 8 \
                                    --num_nodes $SLURM_JOB_NUM_NODES \
                                    --lr_init 0.001
                                    

