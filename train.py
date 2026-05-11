#!/usr/bin/env python
from __future__ import print_function
import numpy as np
from time import time
import os

import h5py
import yaml
import glob
import scipy.interpolate

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau#, LambdaLR
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint,TQDMProgressBar#, Callback

from model import *
from data_generator import *

print(f"PyTorch version: {torch.__version__}", flush=True)

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

        
    #def on_train_epoch_end(self):
    #    train_dataset = self.trainer.datamodule.train_dataset

    def on_train_epoch_start(self):
        self.trainer.datamodule.train_dataset.epoch = self.current_epoch
        print(f"Epoch {self.current_epoch}: updated training dataset epoch counter.", flush=True)
        
    #def on_validation_epoch_end(self):
    #    val_dataset = self.trainer.datamodule.val_dataset




def get_rank() -> int:
    """Best-effort rank detection for SLURM/DDP; defaults to 0."""
    return int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_PROCID", 0)))

def load_train_config(config_path):
    """Load training defaults from YAML config."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    shared = cfg["shared"]
    paths = cfg["paths"]
    training = cfg["training"]

    defaults = {
        # paths
        "data_dir": paths["data_dir"],
        "noise_dir": paths["noise_dir"],
        "checkpoint_dir": paths["checkpoint_dir"],

        # files
        "train_file": training["train_file"],
        "val_file": training["val_file"],

        # training
        "batch_size": training["batch_size"],
        "num_workers": training["num_workers"],
        "lr_init": training["lr_init"],

        # data generation
        "dim": training["dim"],
        "segment_length": training["segment_length"],
        "edge_buffer": training["edge_buffer"],
        "noise_prob": training["noise_prob"],
        "p_higher_init": training["p_higher_init"],
        "p_higher_fin": training["p_higher_fin"],

        # shared preprocessing
        "sample_rate": shared["sample_rate"],
        "band_low": shared["band_low"],
        "band_high": shared["band_high"],
        "bandpass_order": shared["bandpass_order"],
        "psd_floor": shared["psd_floor"],
        "psd_outband": shared["psd_outband"],

        # dataloader mode
        "noise_is_whitened": bool(shared["noise_is_whitened"]),
    }

    return defaults
    
    
def run_early_diagnostics(args, rank: int) -> None:
    """
    Early-run diagnostics:
    - raw strain amplitude stats
    - matched-filter SNR stats

    This reads validation samples lazily from HDF5 instead of loading the
    full validation file into memory.
    """
    if rank != 0:
        return

    val_path = os.path.join(args.data_dir, args.val_file)

    with h5py.File(val_path, "r") as f:
        grp = f["data"]
        H_ds = grp["H1_wave"]
        L_ds = grp["L1_wave"]

        n_total = H_ds.shape[0]

        # ---------- amplitude stats ----------
        n_samples = min(5000, n_total)
        amp_indices = np.linspace(0, n_total - 1, n_samples, dtype=int)

        peak_amps = np.zeros((n_samples,), dtype=np.float64)

        for j, i in enumerate(amp_indices):
            h1 = H_ds[i]
            l1 = L_ds[i]
            peak_amps[j] = max(np.max(np.abs(l1)), np.max(np.abs(h1)))

        print(f"\nRaw strain amplitude stats over {n_samples} validation samples:")
        print(f"  Min:    {np.min(peak_amps):.4e}")
        print(f"  Max:    {np.max(peak_amps):.4e}")
        print(f"  Median: {np.median(peak_amps):.4e}")
        print(f"  Mean:   {np.mean(peak_amps):.4e}")
        print(f"  Std:    {np.std(peak_amps):.4e}")

        # ===================== SNR STATS =====================

        fs = float(args.sample_rate)
        dt = 1.0 / fs

        noise_files = sorted(glob.glob(os.path.join(args.noise_dir, "*.hdf5")))

        if len(noise_files) == 0:
            print(f"\nNo noise HDF5 files found in {args.noise_dir}, skipping SNR stats.")
            return

        noise_path = noise_files[0]

        with h5py.File(noise_path, "r") as nf:
            has_psd = ("psd_L1" in nf) and ("psd_H1" in nf) and ("freqs" in nf)

            if not has_psd:
                print(
                    f"\nNoise file {noise_path} is missing 'psd_L1', "
                    "'psd_H1', or 'freqs'; skipping SNR stats."
                )
                return

            freqs = np.asarray(nf["freqs"][()]).squeeze()
            psd_L_arr = np.asarray(nf["psd_L1"][()]).squeeze()
            psd_H_arr = np.asarray(nf["psd_H1"][()]).squeeze()

            psd_L = scipy.interpolate.interp1d(
                freqs,
                psd_L_arr,
                bounds_error=False,
                fill_value=(psd_L_arr[0], psd_L_arr[-1]),
            )
            psd_H = scipy.interpolate.interp1d(
                freqs,
                psd_H_arr,
                bounds_error=False,
                fill_value=(psd_H_arr[0], psd_H_arr[-1]),
            )

        n_snr = min(2000, n_total)
        snr_indices = np.linspace(0, n_total - 1, n_snr, dtype=int)
        snrs = np.zeros((n_snr,), dtype=np.float64)

        for j, i in enumerate(snr_indices):
            h1 = H_ds[i]
            l1 = L_ds[i]

            snr_L = whiten.matched_filter_snr(l1, psd_L, dt=dt)
            snr_H = whiten.matched_filter_snr(h1, psd_H, dt=dt)
            snrs[j] = np.sqrt(snr_L**2 + snr_H**2)

        print(f"\nMatched-filter SNR stats over {n_snr} validation samples:")
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
        sample_rate=args.sample_rate,
        band_low=args.band_low,
        band_high=args.band_high,
        bandpass_order=args.bandpass_order,
        psd_floor=args.psd_floor,
        psd_outband=args.psd_outband,
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

    try:
        plot_examples(
            X_batch.numpy(),
            y_batch.numpy(),
            snr_batch.numpy(),
            wL_clean.numpy().copy() if wL_clean is not None else None,
            wH_clean.numpy().copy() if wH_clean is not None else None,
            save_path=save_path,
            sample_rate=args.sample_rate,
        )
        print(f"Saved {os.path.abspath(save_path)}", flush=True)
    except Exception as e:
        print(f"[rank {rank}] WARNING: preview plotting failed: {e}", flush=True)
        print("[rank 0] Continuing training without preview plot.", flush=True)    
    
def main():
    
    #### Parse parameters from config or command line ######################
    
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None, help="Optional YAML config file.")
    pre_args, remaining_argv = pre_parser.parse_known_args()
    parser = argparse.ArgumentParser(description="gw detection", parents=[pre_parser])
    
    # Paths (must be set by user, no defaults)
    parser.add_argument('--data_dir', default=None, help='Directory containing the injection HDF5 files')
    parser.add_argument('--noise_dir', default=None, help='Directory containing the noise HDF5 files')
    parser.add_argument('--checkpoint_dir', default=None, help='Output directory for checkpoints, logs, and plots')    
    
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
    parser.add_argument('--sample_rate', type=int, default=4096, help='Sampling rate in Hz; should match the injection and noise files.')
    parser.add_argument('--band_low', type=float, default=25.0, help='Low-frequency cutoff used in dataloader whitening/band-limiting.')
    parser.add_argument('--band_high', type=float, default=450.0, help='High-frequency cutoff used in dataloader whitening/band-limiting.')
    parser.add_argument('--bandpass_order', type=int, default=4, help='Butterworth bandpass filter order used in the dataloader.')
    parser.add_argument('--psd_floor', type=float, default=1e-48, help='Minimum PSD value used for numerical stability.')
    parser.add_argument('--psd_outband', type=float, default=1e40, help='Large PSD value used to suppress frequencies outside the target band.')
    
    # --- dataloader choice: old vs new (equivalent to whether noise is already whitened) ---
    parser.add_argument('--noise_is_whitened', action='store_true', help='Use the OLD dataloader path (expects noise files to already be whitened). If unset, uses the NEW path (raw noise, whitened inside the generator).')
                            
    if pre_args.config is not None:
        parser.set_defaults(**load_train_config(pre_args.config))
    
    args = parser.parse_args(remaining_argv)
    args.config = pre_args.config
    
    missing = [name for name in ["data_dir", "noise_dir", "checkpoint_dir"] if getattr(args, name) is None]
    if missing:
        raise ValueError(
            "Missing required argument(s): "
            + ", ".join(f"--{name}" for name in missing)
            + ". Provide them directly or use --config."
        )    

    os.makedirs(args.checkpoint_dir, exist_ok=True)


    #### Continue with train script ###################
    
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
    print("setting up trainer", flush=True)
    
    num_cuda_devices = torch.cuda.device_count()
    accelerator = "gpu" if num_cuda_devices > 0 else "cpu"
    devices = num_cuda_devices if num_cuda_devices > 0 else 1
    
    use_ddp = accelerator == "gpu" and (devices > 1 or args.num_nodes > 1)
    strategy = "ddp" if use_ddp else "auto"
    
    print(
        f"Trainer config: accelerator={accelerator}, "
        f"devices={devices}, num_nodes={args.num_nodes}, strategy={strategy}",
        flush=True,
    )
    
    trainer = Trainer(
        max_epochs=100,
        num_nodes=args.num_nodes,
        devices=devices,
        accelerator=accelerator,
        strategy=strategy,
        enable_progress_bar=True,
        enable_model_summary=True,
        callbacks=callbacks,
        # limit_train_batches=0.0001,
        # limit_val_batches=0.0001,
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
        sample_rate=args.sample_rate,
        band_low=args.band_low,
        band_high=args.band_high,
        bandpass_order=args.bandpass_order,
        psd_floor=args.psd_floor,
        psd_outband=args.psd_outband,
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
