import numpy as np
import time
import os
import gc

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import scipy
from scipy import signal

from tqdm.auto import tqdm
import h5py
import glob
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import sys
import matplotlib.pyplot as plt

import os
from datetime import datetime, timezone, timedelta
GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc)
GPS_UTC_OFFSET = 18  # seconds


warnings.filterwarnings("ignore", message="Using padding='same' with even kernel lengths and odd dilation may require a zero-padded copy of the input be created")
warnings.filterwarnings("ignore", message="nn.functional.tanh is deprecated. Use torch.tanh instead.")
warnings.filterwarnings("ignore", message="nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.append("/projects/begd/victoria/Wavenet_torch-main/")

from models_torch import *
from data_generators_torch import *

############################# plot dataset #####################
'''def plot_waveforms(wf_dataset, noise_ranges):
    for noise_range in noise_ranges:
        wf_dataset.noise_range = noise_range

        plt.figure(figsize=(20, 8))
        for i in range(3):
            X, y = wf_dataset.__getitem__(i)  
            labels=y[:,0]
            y = np.arange(4096)

            plt.subplot(3, 5, i + 1)
            plt.plot(y, X[:, 0].numpy(), label='L1',linewidth=0.7)
            plt.plot(y, X[:, 1].numpy(), label='H1',linewidth=0.7)
            plt.plot(y, X[:, 2].numpy(), label='V1',linewidth=0.7)
            plt.plot(y, labels, label='label',linewidth=0.7,c='black')
            plt.title(f'Noise Range {noise_range}')
            plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
        
def plot_whitened_waveforms(wf_dataset, noise_ranges):
    # pick some noise file index to get PSDs from
    # (for inspection it doesn't matter much which one)
    psd_L1, psd_H1 = wf_dataset._get_psd_interps_from_noisefile(0)

    for noise_range in noise_ranges:
        wf_dataset.noise_range = noise_range

        plt.figure(figsize=(20, 8))
        for i in range(1):
            # Generate original signal
            X, y = wf_dataset.__getitem__(i)
            labels = y[:, 0]
            t = np.arange(wf_dataset.segment_length)

            # X shape is (segment_length, 2) now: [L1, H1]
            strain_L1 = X[:, 0].numpy()
            strain_H1 = X[:, 1].numpy()

            # Whiten using the new helper (2 channels only)
            strain_whiten_L, strain_whiten_H = whiten.whiten_signal(
                strain_L1, strain_H1, wf_dataset.dt, psd_L1, psd_H1
            )

            # ---- original signal ----
            plt.subplot(2, 1, 1)
            plt.plot(t, strain_L1, label='L1', linewidth=0.7)
            plt.plot(t, strain_H1, label='H1', linewidth=0.7)
            plt.plot(t, labels, label='label', linewidth=0.7, c='black')
            plt.title(f'Original Signal (noise_range={noise_range})')
            plt.legend(loc='upper right')

            # ---- whitened signal ----
            plt.subplot(2, 1, 2)
            plt.plot(t, strain_whiten_L, label='L1 whitened', linewidth=0.7)
            plt.plot(t, strain_whiten_H, label='H1 whitened', linewidth=0.7)
            plt.plot(t, labels, label='label', linewidth=0.7, c='black')
            plt.title('Whitened Signal')
            plt.legend(loc='upper right')

        plt.tight_layout()
        plt.show()


wf_dataset = GWDataset(
    noise_dir='/projects/begd/victoria/Wavenet_torch-main/noise/',
    data_dir='/projects/bbvf/victoria/WaveNet_data/combined_spin/',
    batch_size=32,
    dim=4096,
    n_channels=3,
    shuffle=False,
    train=0,
    gaussian=0,  
    noise_prob=0,
    noise_range=[0.1, 0.3],  
    initial_epoch=1
)'''

########################### inference functions #####################

# --- Reuse training PSD + whitening logic for inference ---------------------

FS_DEFAULT = 4096.0  # Hz

# make a minimal GWDataset-like object just to use its PSD helper
_psd_self = object.__new__(GWDataset)
_psd_self.dt = 1.0 / FS_DEFAULT
# these constants are exactly what GWDataset __init__ sets:
_psd_self.psd_floor   = 1e-48
_psd_self.psd_outband = 1e40
_psd_self.band_low    = 25.0
_psd_self.band_high   = 450.0

def whiten_from_hdf5(fp, fs=FS_DEFAULT):
    """
    Whiten raw H1/L1 strain from a downloader HDF5 using the *same*
    PSD band-limiting + whitening as in GWDataset.__data_generation.
    """
    
    TRUNC = 4096 // 2

    # raw strain (already glitch-cleaned by the downloader)
    strain_L1 = fp["strain_L1"][:]
    strain_H1 = fp["strain_H1"][:]

    # PSD arrays + freqs saved by the downloader
    freqs    = fp["freqs"][:]
    psd_Larr = fp["psd_L1"][:]
    psd_Harr = fp["psd_H1"][:]

    # this calls the exact same function used in training:
    psd_L = GWDataset._make_band_limited_psd(_psd_self, freqs, psd_Larr)
    psd_H = GWDataset._make_band_limited_psd(_psd_self, freqs, psd_Harr)

    dt = 1.0 / fs

    # and this is the same whitening class from data_generators_torch
    wL = whiten.whiten(strain_L1, psd_L, dt)
    wH = whiten.whiten(strain_H1, psd_H, dt)
    
    wL = wL[TRUNC:-TRUNC]
    wH = wH[TRUNC:-TRUNC]

    return wL, wH



GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc)
GPS_UTC_OFFSET = 18  # seconds, valid for O2/O3-era data

def gps_to_utc_datetime(gps_seconds: float) -> datetime:
    """Convert GPS seconds to UTC datetime (valid for modern LIGO runs)."""
    return GPS_EPOCH + timedelta(seconds=gps_seconds - GPS_UTC_OFFSET)

def trigger_time_from_file(filename, trigger_index, sample_rate):
    """
    filename: 'signal_raw_1264622519_1264626615.hdf5'
    trigger_index: 0-based sample index inside the window
    sample_rate: Hz, e.g. 4096
    """
    base = os.path.basename(filename)
    stem, _ = os.path.splitext(base)
    parts = stem.split('_')   # ['signal', 'raw', '1264622519', '1264626615']

    gps_start = float(parts[2])     # window start
    fs = float(sample_rate)

    gps_trigger = gps_start + trigger_index / fs
    utc_trigger = gps_to_utc_datetime(gps_trigger)

    return gps_trigger, utc_trigger



class TimeSeriesDataset(Dataset):
    def __init__(self, data, targets, length, stride=1, start_index=0, end_index=None):
        self.data = data
        self.targets = targets
        self.length = length
        self.stride = stride

        if end_index is None or end_index > len(data):
            end_index = len(data) - 1

        self.start_index = start_index
        self.end_index = end_index

        self.sample_indices = np.arange(self.start_index, self.end_index - self.length + 1, self.stride)

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        start_idx = self.sample_indices[idx]
        end_idx = start_idx + self.length

        # Slice the data and targets for the current sample
        sample_data = self.data[start_idx:end_idx]
        sample_target = self.targets[start_idx:end_idx]
        return sample_data.float(), sample_target.float()

def normalize(strain):
    #std = np.std(strain[:])
    mean=np.mean(strain[:])
    strain[:]+=-mean
    #strain[:] /= std
    return strain

def butter_bandpass_filter(strain, fs=4096, lowcut=10, highcut=1000, order=4, buffer=2048):
    padded_strain = strain#np.pad(strain, (buffer, buffer), mode='constant')
    
    nyq = 0.5 * fs
    b, a = scipy.signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    filtered = scipy.signal.filtfilt(b, a, padded_strain)
    filtered = filtered[buffer:-buffer]
    
    return filtered


# Function to find peaks
def find_peaks(preds, threshold=0.9, width=1000, mean=0.9):
    test_p = preds
    peaks, properties = signal.find_peaks(test_p, height=threshold, width=width, distance=4096 * 1)
    left = properties['left_ips']
    right = properties['right_ips']

    f_left = []
    f_right = []
    for i in range(len(left)):
        sliced = test_p[int(left[i]):int(right[i])]
        if (np.mean(sliced > mean) > 0.5):
            f_left.append(int(left[i]))
            f_right.append(int(right[i]))

    return peaks, f_left, f_right


# Function to merge windows
def merge_windows(triggers_0, triggers_5):
    triggers = {}
    for key in triggers_0.keys():
        right_0 = triggers_0[key]
        right_5 = triggers_5[key]

        combined = right_0.copy()
        for r_5 in right_5:
            keep = True
            for r_0 in right_0:
                if abs(r_5 - r_0) == 1 / 2:
                    keep = False
            if keep:
                combined.append(r_5)

        triggers[key] = combined

    return triggers

def make_preds(whitened_L1, whitened_H1, whitened_V1, model, inference_args,
               dataset_name):
    device = next(model.parameters()).device

    offsets = [0,
               2047 // 2,                # 1 023   ≈ 0.25 s
               2047,                     # 2 047   ≈ 0.50 s  
              2047 + 2047 // 2]         # 3 070   ≈ 0.75 s

    # run a single dataloader through the network
    def _single_pred(dataloader, bar_pos, disable_bar):
        preds = []
        pbar = tqdm(dataloader, leave=True, position=bar_pos,
                    desc=f'Predicting {dataset_name}[{bar_pos}]',
                    dynamic_ncols=True, disable=disable_bar)
        for batch in pbar:
            with torch.no_grad():
                batch = batch[0].to(device)
                probs = model(batch)                    # raw logits
                #probs  = torch.sigmoid(logits)           # convert to [0, 1]
                preds.append(probs.cpu().numpy())
        return np.concatenate(preds).ravel() if preds else np.empty(0)


    # ── normalise & tensor-ise input ───────────────────────────────────────────
    n = min(len(whitened_L1), len(whitened_H1), len(whitened_V1))
    whitened_L1, whitened_H1, whitened_V1 = (
        torch.tensor(normalize(x[:n]).copy(), dtype=torch.float32)
        for x in (whitened_L1, whitened_H1, whitened_V1)
    )
    data = torch.stack((whitened_L1, whitened_H1, whitened_V1), dim=1).to(device)

    # ── build dataloaders for all offsets ──────────────────────────────────────
    loaders = [
        DataLoader(
            TimeSeriesDataset(data, data, length=4096, stride=4096,
                              start_index=off),
            batch_size=inference_args.batch_size, shuffle=False)
        for off in offsets
    ]

    # ── run the model in parallel ──────────────────────────────────────────────
    disable_bars = [False] + [True] * (len(loaders) - 1)
    with ThreadPoolExecutor(max_workers=len(loaders)) as pool:
        futures = [
            pool.submit(_single_pred, dl, i, dis)
            for i, (dl, dis) in enumerate(zip(loaders, disable_bars))
        ]
    preds = [f.result() for f in futures]   

    gc.collect()
    torch.cuda.empty_cache()
    return preds, offsets


def get_triggers(preds_list, offsets, width, threshold,
                 truncation=0, fs=4096):
    """
    Returns {'detection': [t₁, t₂, …]} where each tᵢ is in *seconds*.
    Duplicate triggers arising from overlapping windows are merged
    if they lie closer than 0.25 s (one quarter-window) to a previous one.
    """
    assert len(preds_list) == len(offsets)

    all_triggers = []
    dynamic_mean = max(0, min(0.95, threshold - 0.05))
    #dynamic_mean =  max(0, min(0.85, threshold - 0.10))
    #dynamic_mean=0.5

    for preds, off in zip(preds_list, offsets):
        _, _, right = find_peaks(preds,
                                 threshold=threshold,
                                 width=width,
                                 mean=dynamic_mean)
        # convert sample indices to seconds and apply offset
        all_triggers.extend([(r + truncation + off) / fs for r in right])

    # ── remove duplicates from different windows within 1/4 s  ─────────────
    all_triggers = sorted(all_triggers)
    merged = []
    for t in all_triggers:
        if not merged or t - merged[-1] > 0.25:
            merged.append(t)

    return {'detection': merged}

def trigger_time_from_file(filename, trigger_index, sample_rate):
    """
    filename: e.g. 'signal_raw_1264622519_1264626615.hdf5'
    trigger_index: sample index inside the window (0-based)
    sample_rate: in Hz, e.g. 4096
    """
    base = os.path.basename(filename)
    stem, _ = os.path.splitext(base)          # 'signal_raw_1264622519_1264626615'
    parts = stem.split('_')                   # ['signal', 'raw', '1264622519', '1264626615']

    gps_start = float(parts[2])               # third token = window start GPS
    fs = float(sample_rate)

    gps_trigger = gps_start + trigger_index / fs
    t = Time(gps_trigger, format="gps")
    return gps_trigger, t.to_datetime()       # (GPS float, UTC datetime)


def process_data(data_dir, model, threshold, width, inference_args):
    dataset_name = os.path.splitext(os.path.basename(data_dir))[0]

    with h5py.File(data_dir, 'r') as fp:
        if all(k in fp for k in ("psd_L1", "psd_H1", "freqs")):
            # NEW: raw → whitened using the training pipeline
            whitened_L1, whitened_H1 = whiten_from_hdf5(fp, fs=FS_DEFAULT)
        else:
            # Fallback for old files with pre-whitened strain
            print("[WARN] HDF5 has no PSDs/freqs; using strain_* as-is.")
            whitened_L1 = fp["strain_L1"][:]
            whitened_H1 = fp["strain_H1"][:]

        # third channel for legacy 3-channel models; can be zeros
        whitened_V1 = np.zeros_like(whitened_L1)

    n = min(len(whitened_L1), len(whitened_H1), len(whitened_V1))
    whitened_L1, whitened_H1, whitened_V1 = (
        x[:n] for x in (whitened_L1, whitened_H1, whitened_V1)
    )

    # ── inference ─────────────────────────────────────────────────────────────
    t0 = time.time()
    preds_list, offsets = make_preds(whitened_L1, whitened_H1, whitened_V1,
                                     model, inference_args, dataset_name)
    print(f'Inference   ↯ {time.time() - t0:.1f}s')

    # ── peak finding / trigger extraction ─────────────────────────────────────
    t0 = time.time()
    triggers = get_triggers(preds_list, offsets, width, threshold,
                            truncation=0)
    print(f'Postprocess ↯ {time.time() - t0:.1f}s')

    gc.collect()
    torch.cuda.empty_cache()
    return dataset_name, triggers, whitened_L1, whitened_H1, whitened_V1


    
######################## process and plot triggers ################################################

def tolerant_intersection(*trigger_lists, tolerance=1):
    common_triggers = set(trigger_lists[0])
    for triggers in trigger_lists[1:]:
        common_triggers = {trigger for trigger in common_triggers if any(np.isclose(trigger, t, atol=tolerance) for t in triggers)}
    return list(common_triggers)

def process_triggers(dataset_name_to_dir, dataset_triggers, models, tolerance=0.05, plot=True):
    common_triggers_count = 0
    common_triggers_info = {}
    
    for dataset_name, triggers_dicts in dataset_triggers.items():
        if len(triggers_dicts) == len(models):
            trigger_lists = [triggers_dict['detection'] for triggers_dict in triggers_dicts]
            common_triggers = tolerant_intersection(*trigger_lists, tolerance=tolerance)
            
            for trigger in common_triggers:
                if 1923 <= trigger <= 1927:
                    common_triggers_count += 1
            
            common_triggers_info[dataset_name] = common_triggers
            
            if plot:
                data_dir = dataset_name_to_dir[dataset_name]
                with h5py.File(data_dir, 'r') as fp:
                    strain_L1 = fp['strain_L1'][:]
                    strain_H1 = fp['strain_H1'][:]
                    strain_V1 = fp['strain_L1'][:]
                    
                    min_length = min(len(strain_L1), len(strain_H1))
                    strain_L1 = strain_L1[:min_length]
                    strain_H1 = strain_H1[:min_length]
                    strain_V1 = strain_V1[:min_length]
                    
                plot_results(dataset_name, strain_L1, strain_H1, strain_V1, common_triggers, save_path=f'{dataset_name}.png')
    
    return common_triggers_info, common_triggers_count


            
def plot_results(dataset_name, strain_L1, strain_H1, strain_V1, common_triggers, save_path=None, demo=0):
    strain_L1=normalize(strain_L1)
    strain_H1=normalize(strain_H1)
    strain_V1=normalize(strain_V1)
    plt.figure()
    fig, axs = plt.subplots(3, 1, figsize=(10, 14))
    
    x_vals = np.arange(len(strain_L1)) / 4096
    merger_time=np.size(strain_L1)//2//4096
    print('ground truth merger time:',merger_time)
    

    
    axs[0].plot(x_vals[::4], strain_L1[::4], label='Livingston (L1)')
    axs[0].set_title('Livingston (L1)')
    #axs[0].set_ylim([-7, 7])
    axs[0].legend(loc='upper right')

    axs[1].plot(x_vals[::4], strain_H1[::4], label='Hanford (H1)')
    axs[1].set_title('Hanford (H1)')
    #axs[1].set_ylim([-7, 7])
    axs[1].legend(loc='upper right')


    axs[2].plot(x_vals[::4], np.zeros_like(x_vals[::4]))
    axs[2].axvline(x=merger_time, color='r', linestyle='-', ymin=0.25, ymax=0.75, label='True Signal', linewidth=4)
    for trigger in common_triggers:
        axs[2].axvline(x=trigger, color='g', linestyle='--', ymin=0.0, ymax=1.0, label='Predicted Signal' if trigger == common_triggers[0] else "")
    axs[2].set_title('Predicted Signals')
    axs[2].legend(loc='upper right')

    if demo == 1:
        fig.suptitle(f'Dataset: {dataset_name}, (no ensemble averaging)', fontsize=16)
    else:
        fig.suptitle(f'Dataset: {dataset_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.subplots_adjust(top=0.90)

    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)

    