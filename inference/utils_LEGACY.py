import numpy as np
import time
import os
import gc
import re
import glob

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

from datetime import datetime, timezone

warnings.filterwarnings("ignore", message="Using padding='same' with even kernel lengths and odd dilation may require a zero-padded copy of the input be created")
warnings.filterwarnings("ignore", message="nn.functional.tanh is deprecated. Use torch.tanh instead.")
warnings.filterwarnings("ignore", message="nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.append("/projects/begd/victoria/Wavenet_torch-main")

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
    for noise_range in noise_ranges:
        wf_dataset.noise_range = noise_range

        plt.figure(figsize=(20, 8))
        for i in range(1):
            # Generate original signal
            X, y = wf_dataset.__getitem__(i)  
            labels=y[:,0]
            y = np.arange(4096)

            # Generate whitened signal
            strain_L1 = X[:, 0].numpy()
            strain_H1 = X[:, 1].numpy()
            strain_V1 = X[:, 2].numpy()

            psd_L1 = pickle.load(open(wf_dataset.psd_L1_files, 'rb'), encoding="bytes")
            psd_H1 = pickle.load(open(wf_dataset.psd_H1_files, 'rb'), encoding="bytes")
            psd_V1 = pickle.load(open(wf_dataset.psd_V1_files, 'rb'), encoding="bytes")

            strain_whiten_L, strain_whiten_H, strain_whiten_V = whiten.whiten_signal(
                strain_L1, strain_H1, strain_V1, wf_dataset.dt, psd_L1, psd_H1, psd_V1
            )

            # Plot original signal
            plt.subplot(3, 4, 4 * i + 1)
            plt.plot(y, strain_L1, label='L1',linewidth=0.7)
            plt.plot(y, strain_H1, label='H1',linewidth=0.7)
            plt.plot(y, strain_V1, label='V1',linewidth=0.7)
            plt.plot(y, labels, label='label',linewidth=0.7,c='black')
            plt.title(f'Original Signal')
            plt.legend(loc='upper right')

            # Plot whitened signal
            plt.subplot(3, 4, 4 * i + 2)
            plt.plot(y, strain_whiten_L, label='L1',linewidth=0.7)
            plt.plot(y, strain_whiten_H, label='H1',linewidth=0.7)
            plt.plot(y, strain_whiten_V, label='V1',linewidth=0.7)
            plt.plot(y, labels, label='label',linewidth=0.7,c='black')
            plt.title(f'Whitened Signal')
            plt.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

wf_dataset = WFGDataset(
    noise_dir='/projects/begd/victoria/WaveNet_data/Gaussian_Noise/',
    data_dir='/projects/begd/victoria/WaveNet_data/combined_spin/',
    batch_size=32,
    dim=4096,
    n_channels=3,
    shuffle=False,
    train=0,
    gaussian=1,  
    noise_prob=0,
    noise_range=[0.1, 0.3],  
    initial_epoch=1
)'''

########################### inference functions #####################

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

def normalize(strain): #optional normalization (should be normalized after whitening)
    std = np.std(strain[:])
    strain[:] /= std
    return strain

GPS_UNIX_OFFSET = 315964800

def gps_from_filename(fname):
    m = re.search(r'(\d+)', fname)
    return int(m.group(1)) if m else None

def gps_to_utc_string(gps):
    unix_time = gps + GPS_UNIX_OFFSET
    dt = datetime.fromtimestamp(unix_time, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")



def butter_bandpass_filter(strain, fs=4096, lowcut=10, highcut=1000, order=4, buffer=2048): #optional filter
    padded_strain = np.pad(strain, (buffer, buffer), mode='constant')
    
    nyq = 0.5 * fs
    b, a = scipy.signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    filtered = scipy.signal.filtfilt(b, a, padded_strain)
    filtered = filtered[buffer:-buffer]
    
    return filtered

def make_preds(whitened_L1, whitened_H1, whitened_V1, model, inference_args, dataset_name):
    device = next(model.parameters()).device
    
    def _make_single_pred(dataloader, position,disable_bar):
        preds = []
        pbar = tqdm(dataloader, desc=f'Predicting for {dataset_name}', leave=True, position=position, dynamic_ncols=True, disable=disable_bar)
        for i, inputs in enumerate(pbar):
            with torch.no_grad():
                inputs = inputs[0].to(device)  # Move inputs to GPU
                outputs = model(inputs)
                preds.append(outputs.detach().cpu().numpy())  # Move outputs back to CPU
            pbar.update()
        return np.concatenate(preds).ravel()

    normalized_L1 = normalize(whitened_L1)
    normalized_H1 = normalize(whitened_H1)
    normalized_V1 = normalize(whitened_V1)

    if inference_args.length is not None:
        whitened_L1 = torch.tensor(whitened_L1[:length])
        whitened_H1 = torch.tensor(whitened_H1[:length])
        whitened_V1 = torch.tensor(whitened_V1[:length])
    else:
        whitened_L1 = torch.tensor(whitened_L1)
        whitened_H1 = torch.tensor(whitened_H1)
        whitened_V1 = torch.tensor(whitened_V1)

    # --- enforce equal lengths across detectors ---
    min_len = min(len(whitened_L1), len(whitened_H1), len(whitened_V1))
    whitened_L1 = whitened_L1[:min_len]
    whitened_H1 = whitened_H1[:min_len]
    whitened_V1 = whitened_V1[:min_len]

    data = torch.stack((whitened_L1, whitened_H1, whitened_V1), dim=1).to(device)  # Move data to GPU

    dataloader_0 = TimeSeriesDataset(data=data, targets=data, length=4096, stride=4096, start_index=0)
    dataloader_5 = TimeSeriesDataset(data=data, targets=data, length=4096, stride=4096, start_index=2047)

    dataloader_0 = DataLoader(dataloader_0, batch_size=inference_args.batch_size, shuffle=False)
    dataloader_5 = DataLoader(dataloader_5, batch_size=inference_args.batch_size, shuffle=False)

    model.eval()
    
    disable_bar_0 = False if device == torch.device('cuda:0') or device == torch.device('cuda') else True
    disable_bar_5 = True

    with ThreadPoolExecutor(max_workers=8) as executor:  # Allocate 10 workers for inner level
        future_preds_0 = executor.submit(_make_single_pred, dataloader_0, 0, disable_bar_0)
        future_preds_5 = executor.submit(_make_single_pred, dataloader_5, 1, disable_bar_5)

        preds_0 = future_preds_0.result()
        preds_5 = future_preds_5.result()

    return preds_0, preds_5

# Function to find peaks
def find_peaks(preds, threshold=0.9, width=[1500, 3000], mean=0.9):
    test_p = preds
    peaks, properties = signal.find_peaks(test_p, height=threshold, width=width, distance=4096 * 1)
    left = properties['left_ips']
    right = properties['right_ips']

    f_left = []
    f_right = []
    for i in range(len(left)):
        sliced = test_p[int(left[i]):int(right[i])]
        if (np.mean(sliced > mean) > mean):
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

# Convert right ips to GPS times and merge the two windows
def get_triggers(preds_0, preds_5, width, threshold, truncation=0, window_shift=2048):
    triggers_0, triggers_5 = {}, {}
    key = 'detection'

    dynamic_mean = max(0, min(0.95, threshold - 0.05))

    peak_0, left_0, right_0 = find_peaks(preds_0, threshold=threshold, width=[width, 3000], mean=dynamic_mean)
    peak_5, left_5, right_5 = find_peaks(preds_5, threshold=threshold, width=[width, 3000], mean=dynamic_mean)

    triggers_0[key] = [(x + truncation) / 4096 for x in right_0]
    triggers_5[key] = [(x + truncation + window_shift) / 4096 for x in right_5]

    # Merge the two windows
    triggers = merge_windows(triggers_0, triggers_5)

    return triggers

def process_data(data_dir, model, threshold, width, inference_args):
    #dataset_name=data_dir.split('/')[-1].split('_')[0]
    #dataset_name = os.path.splitext(os.path.basename(data_dir))[0]
    dataset_name = os.path.basename(data_dir)  # keep the .hdf5


    for i in [1]:
        # Load strains
        with h5py.File(data_dir, 'r') as fp:
            strain_L1 = fp['strain_L1'][:]
            strain_H1 = fp['strain_H1'][:]
            strain_V1 = np.zeros_like(strain_L1)

        # Make preds
        start_time = time.time()
        preds_0, preds_5 = make_preds(strain_L1, strain_H1, strain_V1, model, inference_args, dataset_name)
        elapsed_time = time.time() - start_time

        # Post Process and find triggers
        start_time = time.time()
        triggers = get_triggers(preds_0, preds_5, width, threshold, truncation=0, window_shift=2048)
        elapsed_time = time.time() - start_time

        return dataset_name, triggers, strain_L1, strain_H1, strain_V1

    
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
                if 1921 <= trigger <= 1930:
                    common_triggers_count += 1
            
            common_triggers_info[dataset_name] = common_triggers
            
            data_dir = dataset_name_to_dir[dataset_name]
            with h5py.File(data_dir, 'r') as fp:
                strain_L1 = fp['strain_L1'][:]
                strain_H1 = fp['strain_H1'][:]
                strain_V1 = fp['strain_V1'][:]

            if plot==True:
                plot_results(dataset_name, strain_L1, strain_H1, strain_V1, common_triggers, save_path=f'{dataset_name}.png')
    
    return common_triggers_info, common_triggers_count


            
def plot_results(dataset_name, strain_L1, strain_H1, strain_V1, common_triggers, save_path=None, demo=0):
    plt.figure()
    fig, axs = plt.subplots(4, 1, figsize=(10, 14))
    
    x_vals = np.arange(len(strain_L1)) / 4096

    axs[0].plot(x_vals[::4], strain_L1[::4], label='Livingston (L1)')
    axs[0].set_title('Livingston (L1)')
    axs[0].set_ylim([-7, 7])
    axs[0].legend(loc='upper right')

    axs[1].plot(x_vals[::4], strain_H1[::4], label='Hanford (H1)')
    axs[1].set_title('Hanford (H1)')
    axs[1].set_ylim([-7, 7])
    axs[1].legend(loc='upper right')

    axs[2].plot(x_vals[::4], strain_V1[::4], label='Virgo (V1)')
    axs[2].set_title('Virgo (V1)')
    axs[2].set_ylim([-7, 7])
    axs[2].legend(loc='upper right')

    axs[3].plot(x_vals[::4], np.zeros_like(x_vals[::4]))
    axs[3].axvline(x=1925, color='r', linestyle='-', ymin=0.25, ymax=0.75, label='True Signal', linewidth=4)
    for trigger in common_triggers:
        axs[3].axvline(x=trigger, color='g', linestyle='--', ymin=0.0, ymax=1.0, label='Predicted Signal' if trigger == common_triggers[0] else "")
    axs[3].set_title('Predicted Signals')
    axs[3].legend(loc='upper right')

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

    
