#!/usr/bin/env python
from __future__ import print_function
import numpy as np
from time import time
import os
import sys
import gc
import cProfile
import math

import h5py
import glob
import scipy.interpolate

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau#, LambdaLR
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint,TQDMProgressBar#, Callback
from pytorch_lightning.loggers import TensorBoardLogger
#from torch.utils.data.distributed import DistributedSampler

from model import *
from data_generator import *

print(torch.__version__)

import argparse

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')

class LightningModel(LightningModule):
    def __init__(self, lr=0.001, internal_epoch=0):
        super(LightningModel, self).__init__()
        self.model = full_module()
        self.lr = lr
        self.internal_epoch = internal_epoch
        self.criterion = nn.BCELoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            threshold=3e-4,       
            threshold_mode='abs',
            min_lr=1e-6,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  
                'interval': 'epoch',  
                'frequency': 1,  
                'strict': True,
                'reduce_on_plateau': True,
            }
        }

    def training_step(self, batch, batch_idx):
        inputs, targets = batch  
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)   # standard BCE
        return loss

        
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch  
        
        with torch.no_grad():
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

        self.log(
            'val_loss',
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

        
    def on_train_epoch_end(self):
        train_dataset = self.trainer.datamodule.train_dataset

    def on_train_epoch_start(self):
        self.trainer.datamodule.train_dataset.epoch = self.current_epoch
        print(f"[DEBUG] LightningModel → Epoch {self.current_epoch} — train_dataset.epoch = {self.trainer.datamodule.train_dataset.epoch}")

        
    def on_validation_epoch_end(self):
        val_dataset = self.trainer.datamodule.val_dataset




def get_rank() -> int:
    """Best-effort rank detection for SLURM/DDP; defaults to 0."""
    return int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_PROCID", 0)))


def run_early_diagnostics(args, rank: int) -> None:
    """
    Early-run diagnostics:
    - raw strain amplitude stats
    - SNR stats
    """
    if rank != 0:
        return

    data_dir = args.data_dir
    val_path = os.path.join(args.data_dir, args.val_file)

    with h5py.File(val_path, "r") as f:
        grp = f["data"]
        H = grp["H1_wave"][()]
        L = grp["L1_wave"][()]

    # ---------- amplitude stats ----------
    n_samples = min(5000, H.shape[0])
    peak_amps = np.zeros((n_samples,))

    for i in range(n_samples):
        h1 = H[i]
        l1 = L[i]
        peak_amps[i] = max(np.max(np.abs(l1)), np.max(np.abs(h1)))

    print(f"\n Raw strain amplitude stats over {n_samples} samples:")
    print(f"  Min:    {np.min(peak_amps):.4e}")
    print(f"  Max:    {np.max(peak_amps):.4e}")
    print(f"  Median: {np.median(peak_amps):.4e}")
    print(f"  Mean:   {np.mean(peak_amps):.4e}")
    print(f"  Std:    {np.std(peak_amps):.4e}")

    # ===================== SNR STATS  =====================
    
    fs = 4096.0
    dt = 1.0 / fs
    
    noise_dir = args.noise_dir
    noise_files = sorted(glob.glob(os.path.join(noise_dir, "*.hdf5")))
    
    if len(noise_files) == 0:
        print(f"\nNo noise HDF5 files found in {noise_dir}, skipping SNR stats.")
    else:
        noise_path = noise_files[0]
        with h5py.File(noise_path, "r") as nf:
            has_psd = ("psd_L1" in nf) and ("psd_H1" in nf) and ("freqs" in nf)
            if not has_psd:
                print(f"\nNoise file {noise_path} is missing 'psd_L1', 'psd_H1', or 'freqs'; skipping SNR stats.")
            else:
                freqs = np.asarray(nf["freqs"][()]).squeeze()
                psd_L_arr = np.asarray(nf["psd_L1"][()]).squeeze()
                psd_H_arr = np.asarray(nf["psd_H1"][()]).squeeze()
    
                psd_L = scipy.interpolate.interp1d(
                    freqs, psd_L_arr,
                    bounds_error=False,
                    fill_value=(psd_L_arr[0], psd_L_arr[-1])
                )
                psd_H = scipy.interpolate.interp1d(
                    freqs, psd_H_arr,
                    bounds_error=False,
                    fill_value=(psd_H_arr[0], psd_H_arr[-1])
                )
    
                n_snr = min(2000, H.shape[0])
                snrs = np.zeros((n_snr,), dtype=np.float64)
    
                for i in range(n_snr):
                    h1 = H[i]
                    l1 = L[i]
    
                    snr_L = whiten.matched_filter_snr(l1, psd_L, dt=dt)
                    snr_H = whiten.matched_filter_snr(h1, psd_H, dt=dt)
                    snrs[i] =  np.sqrt(snr_L**2 + snr_H**2)
    
                print(f"\nMatched-filter SNR stats over {n_snr} validation signals:")
                print(f"  Min SNR:    {np.min(snrs):.3f}")
                print(f"  Max SNR:    {np.max(snrs):.3f}")
                print(f"  Median SNR: {np.median(snrs):.3f}")
                print(f"  Mean SNR:   {np.mean(snrs):.3f}")
                print(f"  Std SNR:    {np.std(snrs):.3f}")



def plot_samples(args, rank: int) -> None:
    """Plot a batch of examples."""
    if rank != 0:
        return
    
    if args.noise_is_whitened:
        print("[rank 0] skipping plotting because noise_is_whitened=True (old dataloader path)", flush=True)
        return

    print(f'[rank {rank}] plotting waveforms', flush=True)

    plot_dm = WaveformDataModule(
        noise_dir=args.noise_dir,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        noise_prob=0.3,
        num_workers=args.num_workers,
        p_higher_init=0.5,
        p_higher_fin=0.1,
        segment_length=args.segment_length,
        edge_buffer=args.edge_buffer,
        dim=args.dim,
        train_file=args.train_file,
        val_file=args.val_file,
        noise_is_whitened=args.noise_is_whitened,
    )


    plot_dm.setup(stage='fit')
    dataset = plot_dm.train_dataset
    dataset.plotsamples = True

    plain_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=plot_dm.batch_size,
        shuffle=True,
        num_workers=0,
    )

    X_batch, y_batch, snr_batch, wL_clean, wH_clean = next(iter(plain_loader))

    save_path = os.path.join(args.checkpoint_dir, "training_batch_preview.png")

    plot_examples(
        X_batch.numpy(),
        y_batch.numpy(),
        snr_batch.numpy(),
        wL_clean.numpy().copy(),
        wH_clean.numpy().copy(),
        save_path=save_path,
    )
    print(f"Saved {os.path.abspath(save_path)}", flush=True)
    
    
def main():
    parser = argparse.ArgumentParser(description="gw detection")

    # Paths (must be set by user, no defaults)
    parser.add_argument('--data_dir', required=True, help='Directory containing the injection HDF5 files')
    parser.add_argument('--noise_dir', required=True, help='Directory containing the noise HDF5 files')
    parser.add_argument('--checkpoint_dir', required=True, help='Output directory for checkpoints, logs, and plots')
    
    # --- injection file names  ---
    parser.add_argument('--train_file', type=str, default='train.hdf', help='train injection file name inside data_dir')
    parser.add_argument('--val_file', type=str, default='val.hdf', help='val injection file name inside data_dir')
    
    # --- training configuration ---
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers in dataloader')
    parser.add_argument('--num_nodes', type=int, default=1, help='number of nodes')
    parser.add_argument('--lr_init', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--initial_epoch', type=int, default=0, help='starting epoch counter (for curricula / schedules; no checkpoint resume)')
    
    # Data-generation 
    parser.add_argument('--noise_prob', type=float, default=0.6, help='probability of sampling noise-only examples')
    parser.add_argument('--p_higher_init', type=float, default=0.9, help='initial probability of sampling higher snr range')
    parser.add_argument('--p_higher_fin', type=float, default=0.25, help='final probability of sampling higher snr range')
    parser.add_argument('--segment_length', type=int, default=4096, help='segment length fed to the model')
    parser.add_argument('--dim', type=int, default=1024, help='how many samples before merger at labeled as belonging to the signal class')
    parser.add_argument('--edge_buffer', type=int, default=2048, help='Number of samples trimmed from each side after whitening/filtering to reduce edge artifacts.')
    
    # --- dataloader choice: old vs new (equivalent to whether noise is already whitened) ---
    parser.add_argument('--noise_is_whitened', action='store_true',
                        help='Use the OLD dataloader path (expects noise files to already be whitened). If unset, uses the NEW path (raw noise, whitened inside the generator).')
                        
    args = parser.parse_args()
    

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    rank = get_rank()

    #----- callbacks --------
    RegularModelCheckpoints = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='model_attenGW-{epoch:02d}-{val_loss:.5f}',
        monitor='val_loss',
        mode='min',
        save_top_k=-1,
    )
    progress_bar = TQDMProgressBar(refresh_rate=100)
    callbacks = [RegularModelCheckpoints, LearningRateMonitor(logging_interval='epoch'), progress_bar]

    #----- set up training -----
    print('setting up trainer')
    devices = torch.cuda.device_count()
    accelerator = "gpu" if devices > 0 else "cpu"
    devices = devices if devices > 0 else 1
    strategy = "ddp" if accelerator == "gpu" and (devices > 1 or args.num_nodes > 1) else "auto"
    
    trainer = Trainer(
        max_epochs=100,
        num_nodes=args.num_nodes,
        devices=devices,
        accelerator=accelerator,
        strategy=strategy,
        enable_progress_bar=True,
        enable_model_summary=True,
        callbacks=callbacks,
    )


    model = LightningModel(lr=args.lr_init, internal_epoch=args.initial_epoch)

    #------ DataModule  ------
    data_module = WaveformDataModule(
        noise_dir=args.noise_dir,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        dim=args.dim,
        segment_length=args.segment_length,
        edge_buffer=args.edge_buffer,
        noise_prob=args.noise_prob,
        num_workers=args.num_workers,
        p_higher_init=args.p_higher_init,
        p_higher_fin=args.p_higher_fin,
        train_file=args.train_file,
        val_file=args.val_file,
        noise_is_whitened=args.noise_is_whitened,
    )

    # Early-run diagnostics and plotting (rank 0 only)
    run_early_diagnostics(args, rank)
    plot_samples(args, rank)

    #train
    t0 = time()
    trainer.fit(model, datamodule=data_module)
    t1 = time()

    print('**Evaluation time: %s' % (t1 - t0))


if __name__ == '__main__':
    main()
