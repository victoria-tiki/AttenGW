#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4   #match with gpus-per-node
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS
#SBATCH --partition=ghx4       # <- or one of: cpu, gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=begd-dtai-gh
#SBATCH --job-name=AttenGW
#SBATCH --time=20:00:00      # hh:mm:ss for the job
#SBATCH --constraint="projects"
### GPU options ###
#SBATCH --gpus-per-node=4


module load python
module load python/miniforge3_pytorch/2.5.0
module load cray-hdf5/1.14.3.1
pip install --user --upgrade --force-reinstall h5py
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