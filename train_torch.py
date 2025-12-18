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
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data.distributed import DistributedSampler
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import Callback,TQDMProgressBar
from torch.optim.lr_scheduler import LambdaLR


from models_torch import *
from data_generators_torch import *


print(torch.__version__)

import argparse

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')



class LightningModel(LightningModule):
    def __init__(self, lr=0.001, internal_epoch=0,pos_weight=2.0):
        super(LightningModel, self).__init__()
        self.model = full_module()
        self.lr = lr
        self.internal_epoch = internal_epoch
        self.pos_weight = pos_weight   # NEW
        self.criterion = nn.BCELoss()
        
        #if torch.cuda.is_available():
        #    self.model = self.model.to("cuda")

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

    '''def validation_step(self, batch, batch_idx):
        inputs, targets = batch  
        with torch.no_grad():
            outputs = self.model(inputs)
            #pos_w = torch.tensor([self.pos_weight], device=outputs.device)
            #loss = F.binary_cross_entropy_with_logits(
            #            outputs, targets, pos_weight=pos_w
            #        )
            loss = self.criterion(outputs, targets)   # standard BCE
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss'''
        
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



def main():
    parser = argparse.ArgumentParser(description="gw detection")
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--data_dir', help='root directory', default='/home/victoria/WaveNet_data/precessing_test/')
    parser.add_argument('--checkpoint_dir', help='root directory', default='/home/victoria/WaveNet_training/spin_precessing/bns/checkpoint/')
    parser.add_argument('--noise_dir', help='noise directory', default='/home/victoria/WaveNet_data/noise/')
    parser.add_argument('--num_workers', type=int, help='number of workers in dataloader', default=1)
    parser.add_argument('--num_nodes', type=int, help='number of nodes', default=1)
    parser.add_argument('--lr_init', type=float, help='initial learning rate', default=0.001)
    parser.add_argument('--initial_epoch', type=int, help='Initial epoch for training continuation', default=0)
    parser.add_argument('--plot_sample_waveforms', type=bool, help='whether or not to plot sample waveforms', default=True)
    parser.add_argument('--resume_checkpoint', type=str, help='checkpoint filename to resume from (relative to checkpoint_dir)', default='model.ckpt')

    args = parser.parse_args()

    def get_rank():
        return int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_PROCID", 0)))
    
    #----- callbacks --------v
    RegularModelCheckpoints = ModelCheckpoint(dirpath=args.checkpoint_dir, filename='model_attenGW_morenoise_{epoch:02d}-{val_loss:.5f}', monitor='val_loss', mode='min', save_top_k=-1)
    progress_bar = TQDMProgressBar(refresh_rate=100)
    
 patience_after_reset=3)
    callbacks = [RegularModelCheckpoints, LearningRateMonitor(logging_interval='epoch'), progress_bar]

    #----- set up training -----
    print('setting up trainer')
    devices = torch.cuda.device_count()
    devices = devices if devices != 0 else 4 

    trainer = Trainer(
        max_epochs=100,
        num_nodes=args.num_nodes,
        devices=devices,
        accelerator="gpu",
        strategy="ddp",
        #limit_train_batches=0.001,  
        #limit_val_batches=0.001,    
        enable_progress_bar=True,
        enable_model_summary=True,
        callbacks=callbacks,
    )
        
        
        #------ define training from scratch or checkpoint ------
    if args.initial_epoch > 0 and args.resume_checkpoint is not None:
        print(f"Resuming from checkpoint: {args.resume_checkpoint}")
        checkpoint_path = os.path.join(args.checkpoint_dir, args.resume_checkpoint)
        model = LightningModel.load_from_checkpoint(checkpoint_path,lr=args.lr_init,internal_epoch=args.initial_epoch)

        resumed_checkpoint_name = f"resumed_{args.resume_checkpoint}"
        print(f"Future checkpoints will be saved as: {resumed_checkpoint_name}")
    else:
        print('starting from scratch')
        model = LightningModel(lr=args.lr_init, internal_epoch=args.initial_epoch)

        
    
    rank = get_rank()
    if rank == 0:   
        
        data_dir = args.data_dir
        val_path = os.path.join(data_dir, "test_500.hdf")

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
        
        print(f"\n📊 Raw strain amplitude stats over {n_samples} samples:")
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
            print(f"\n⚠️ No noise HDF5 files found in {noise_dir}, skipping SNR stats.")
        else:
            noise_path = noise_files[0]
            with h5py.File(noise_path, "r") as nf:
                has_psd = ("psd_L1" in nf) and ("psd_H1" in nf) and ("freqs" in nf)
                if not has_psd:
                    print(
                        f"\n⚠️ Noise file {noise_path} is missing 'psd_L1', 'psd_H1', or 'freqs'; "
                        "skipping SNR stats."
                    )
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
        
                    print(f"\n📈 Matched-filter SNR stats over {n_snr} validation signals:")
                    print(f"  Min SNR:    {np.min(snrs):.3f}")
                    print(f"  Max SNR:    {np.max(snrs):.3f}")
                    print(f"  Median SNR: {np.median(snrs):.3f}")
                    print(f"  Mean SNR:   {np.mean(snrs):.3f}")
                    print(f"  Std SNR:    {np.std(snrs):.3f}")


    #------- plot some samples -------
    if args.plot_sample_waveforms and rank==0:
        print(f'[rank {rank}] plotting waveforms', flush=True)
        
        data_module = WaveformDataModule(
            noise_dir=args.noise_dir,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            noise_prob=0.3,
            num_workers=args.num_workers,
            p_higher_init=0.5,
            p_higher_fin=0.1
        )
        
        data_module.setup(stage='fit')                   
        dataset = data_module.train_dataset           
        dataset.plotsamples = True
        plain_loader = torch.utils.data.DataLoader(   
            dataset,
            batch_size = data_module.batch_size,
            shuffle    = True,       
            num_workers= 0           
        )
        X_batch, y_batch, snr_batch, wL_clean, wH_clean = next(iter(plain_loader))
        
        plot_examples(
            X_batch.numpy(),
            y_batch.numpy(),
            snr_batch.numpy(),
            wL_clean.numpy().copy(),
            wH_clean.numpy().copy(),
            save_path="training_batch_preview.png"
        )
        print("Saved training_batch_preview.png")
        
    
    #----- define dataloders -----    
    print('defining dataloaders', flush=True)
    data_module = WaveformDataModule(
        noise_dir=args.noise_dir,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        #split_ratio=0.8,
        noise_prob=0.60,
        num_workers=args.num_workers,
        p_higher_init=0.9,
        p_higher_fin=0.25
    )
    

    
    #train
    t0 = time()
    trainer.fit(model, datamodule=data_module)
    t1 = time()

    print('**Evaluation time: %s' % (t1 - t0))


if __name__ == '__main__':
    main()
