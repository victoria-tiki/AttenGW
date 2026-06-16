import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import pickle
import os
import glob
import scipy.interpolate
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

from pytorch_lightning import LightningDataModule
from torch.utils.data.distributed import DistributedSampler



class whiten:
    @staticmethod
    def whiten(strain, interp_psd, dt, floor=1e-48):
        Nt = len(strain)
        freqs = np.fft.rfftfreq(Nt, dt)
        hf = np.fft.rfft(strain)

        psd_vals = interp_psd(freqs)

        psd_vals = np.maximum(psd_vals, floor)

        norm = 1./np.sqrt(1./(dt*2))
        white_hf = hf / np.sqrt(psd_vals) * norm
        white_ht = np.fft.irfft(white_hf, n=Nt)
        return white_ht
        



    @staticmethod
    def whiten_signal(strain_L1, strain_H1, dt, psd_L1, psd_H1):
        strain_whiten_L = whiten.whiten(strain_L1, psd_L1, dt)
        strain_whiten_H = whiten.whiten(strain_H1, psd_H1, dt)

        strain_whiten_L /= np.amax(np.absolute(strain_whiten_L))
        strain_whiten_H /= np.amax(np.absolute(strain_whiten_H))

        return strain_whiten_L, strain_whiten_H

    @staticmethod
    def get_whitened_ligo_noise_chunk(strain, noise_strain_fn, gaussian=0):
        with h5py.File(noise_strain_fn, 'r') as f:
            strain_L1 = f['strain_L1']
            strain_H1 = f['strain_H1']

            starting_index = np.random.randint(0, len(strain_H1)-len(strain))
            ligo_noise_L = strain_L1[starting_index:starting_index+len(strain)]
            ligo_noise_H = strain_H1[starting_index:starting_index+len(strain)]

        return ligo_noise_L, ligo_noise_H

    @staticmethod
    def mix_signal_and_noise(strain_whiten_L, strain_whiten_H, ligo_noise_L, ligo_noise_H, noise_range):
        target_std = np.random.uniform(noise_range[0], noise_range[1])

        ligo_noise_whiten_L = target_std * (ligo_noise_L / np.std(ligo_noise_L))
        ligo_noise_whiten_H = target_std * (ligo_noise_H / np.std(ligo_noise_H))

        mixed_L = (strain_whiten_L + ligo_noise_whiten_L) / np.std(strain_whiten_L + ligo_noise_whiten_L)
        mixed_H = (strain_whiten_H + ligo_noise_whiten_H) / np.std(strain_whiten_H + ligo_noise_whiten_H)

        return mixed_L, mixed_H

#original 4096 file segment, with possibility of being moved outside the window

    @staticmethod
    def load_interp_psd(path):
        psd_obj = pickle.load(open(path, "rb"), encoding="bytes")
    
        # Case 1: already an interp1d
        if isinstance(psd_obj, scipy.interpolate.interp1d):
            return psd_obj
    
        # Case 2: legacy list wrapper [interp1d]
        if isinstance(psd_obj, list) and len(psd_obj) > 0 and isinstance(psd_obj[0], scipy.interpolate.interp1d):
            return psd_obj[0]
    
        # Case 3: (freqs, psd_vals) or [freqs, psd_vals]
        if isinstance(psd_obj, (tuple, list)) and len(psd_obj) == 2:
            freqs, vals = psd_obj
            freqs = np.asarray(freqs)
            vals  = np.asarray(vals)
            return scipy.interpolate.interp1d(
                freqs, vals, bounds_error=False, fill_value=(vals[0], vals[-1])
            )
    
        # Case 4: dict-like {"freqs":..., "psd":...} 
        if isinstance(psd_obj, dict):
            key_map = {k.decode() if isinstance(k, (bytes, bytearray)) else k: k for k in psd_obj.keys()}
            def get_any(names):
                for n in names:
                    if n in key_map:
                        return psd_obj[key_map[n]]
                return None
    
            freqs = get_any(["freqs", "f", "frequency", "frequencies"])
            vals  = get_any(["psd", "S_n", "Sn", "values", "asd2"])
            if freqs is not None and vals is not None:
                freqs = np.asarray(freqs)
                vals  = np.asarray(vals)
                return scipy.interpolate.interp1d(
                    freqs, vals, bounds_error=False, fill_value=(vals[0], vals[-1])
                )
    
        raise TypeError(
            f"Unexpected PSD format in file: {path}. "
            f"Got type {type(psd_obj)} with repr {repr(psd_obj)[:200]}"
        )


    @staticmethod
    def matched_filter_snr(h, psd, dt, floor=1e-48):
        Nt = len(h)
    
        if Nt > 1:
            win = np.hanning(Nt)
            h_win = h * win
        else:
            h_win = h
    
        freqs = np.fft.rfftfreq(Nt, dt)
        hf = np.fft.rfft(h_win)
        psd_vals = psd(freqs)
        psd_vals = np.maximum(psd_vals, floor)
    
        df = freqs[1] - freqs[0]  
        rho2 = 4.0 * (dt**2) * np.sum((np.abs(hf) ** 2) / psd_vals) * df
    
        rho2 = float(np.real(rho2))
        if rho2 < 0:
            rho2 = 0.0
        return np.sqrt(rho2)

        
def low_max_snr(epoch):
    """Legacy SNR/noise-std schedule used only when noise_is_whitened=True."""
    ranges = [
        [0.0, 0.3],
        [0.0, 0.6],
        [0.0, 0.9],
        [0.3, 1.2],
        [0.3, 1.5],
        [0.6, 1.8],
        [0.6, 2.0],
        [0.9, 2.0],
        [1.0, 2.0],
        [0.6, 2.0],
    ]

    boundaries = [2, 4, 6, 8, 10, 12, 14, 16, 28]
    idx = np.searchsorted(boundaries, epoch, side="left")
    return ranges[idx]       
    


def plot_examples(X_batch, y_batch, snrs, wL_clean=None, wH_clean=None,
                  save_path="training_examples.png", sample_rate=4096):

    snrs = np.asarray(snrs)

    n_plot = min(5, len(X_batch), len(y_batch), len(snrs))

    if wL_clean is not None:
        n_plot = min(n_plot, len(wL_clean))
    if wH_clean is not None:
        n_plot = min(n_plot, len(wH_clean))

    fig, axs = plt.subplots(n_plot, 2, figsize=(14, 2.8 * n_plot), sharex=False)

    if n_plot == 1:
        axs = np.expand_dims(axs, axis=0)

    for i in range(n_plot):
        signal_L = X_batch[i,:,0]
        signal_H = X_batch[i,:,1]
        target = y_batch[i, :, 0]

        rho_vals = np.asarray(snrs[i]).reshape(-1)
        rho = float(rho_vals[0]) if len(rho_vals) > 0 else 0.0

        times = np.arange(len(signal_H)) / sample_rate
        
        if rho < 0.01:
            t_min = 0
            t_max = 1.0
        else:
            merger_indices = np.where(target > 0)[0]
            merger_idx = merger_indices[-1]
            peak_time = times[merger_idx]
            t_min = peak_time - 0.3
            t_max = peak_time + 0.1

        # Plot H1
        axs[i, 0].plot(times, signal_H, label="H1 signal + noise",
                       color="#1f77b4", lw=1.1)
        if wH_clean is not None:
            axs[i, 0].plot(times, wH_clean[i], label="Injected signal",
                           color="#ff7f0e", lw=1.6, alpha=0.9)

        ymin, ymax = axs[i, 0].get_ylim()
        axs[i, 0].fill_between(times, ymin, ymax, where=target > 0,
                               color="#9ecae1", alpha=0.35,
                               label="Positive label region")

        axs[i, 0].set_ylabel("H1 strain", fontsize=13)
        axs[i, 0].set_title(f"SNR = {rho:.2f}", fontsize=15, pad=8)
        axs[i, 0].set_xlim([t_min, t_max])
        axs[i, 0].grid(True, alpha=0.25)

        # Plot L1
        axs[i, 1].plot(times, signal_L, label="L1 signal + noise",
                       color="#1f77b4", lw=1.1)
        if wL_clean is not None:
            axs[i, 1].plot(times, wL_clean[i], label="Injected signal",
                           color="#ff7f0e", lw=1.6, alpha=0.9)

        ymin, ymax = axs[i, 1].get_ylim()
        axs[i, 1].fill_between(times, ymin, ymax, where=target > 0,
                               color="#9ecae1", alpha=0.35,
                               label="Positive label region")

        axs[i, 1].set_ylabel("L1 strain", fontsize=13)
        axs[i, 1].set_title(f"SNR = {rho:.2f}", fontsize=15, pad=8)
        axs[i, 1].set_xlim([t_min, t_max])
        axs[i, 1].grid(True, alpha=0.25)

        axs[i, 0].set_xlabel("Time [s]", fontsize=13)
        axs[i, 1].set_xlabel("Time [s]", fontsize=13)

        axs[i, 0].tick_params(axis="both", labelsize=11)
        axs[i, 1].tick_params(axis="both", labelsize=11)
        
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()



                          
class GWDataset(Dataset):

    def __init__(self, noise_dir, data_dir, batch_size=32, dim=2048, n_channels=2,
             shuffle=True, train=True, gaussian=False, noise_prob=0.6,
             initial_epoch=1, segment_length=4096, edge_buffer=2048, merger_out_prob=0.2,
             validation_epoch=None, p_higher_init=0.5, p_higher_fin=0.05, snr_range_high=(10.0, 25.0), snr_range_low=(7.0, 15.0),
             train_file="train.hdf", val_file="test.hdf",
             noise_is_whitened=False, noise_range=None,
             sample_rate=4096, band_low=25.0, band_high=450.0,
             bandpass_order=4, psd_floor=1e-48, psd_outband=1e40):

        self.noise_dir = noise_dir
        self.segment_length = segment_length
        self.label_width = dim
        self.train_file = train_file
        self.val_file = val_file
        self.noise_is_whitened = noise_is_whitened
        self.noise_range = noise_range
        
        # ============================================================
        # train=True  -> use train.hdf5
        # train=False -> use val.hdf5
        # ============================================================
        if train:
            data_path = os.path.join(data_dir, self.train_file)
        else:
            data_path = os.path.join(data_dir, self.val_file)
            
        self.data_files = [data_path]
        self.file_handlers = [h5py.File(data_path, "r")]

        grp = self.file_handlers[0]["data"]
        num_samples = grp["H1_wave"].shape[0]  # (N, 8192)
        self.indices = [(0, i) for i in range(num_samples)]


        self.gaussian = gaussian
        if gaussian:
            self.psd_L1_files = noise_dir + 'Gaussian_Noise/psd_L.pkl'
            self.psd_H1_files = noise_dir + 'Gaussian_Noise/psd_H.pkl'
            self.noise_files = sorted(glob.glob(noise_dir + 'Gaussian_Noise/gaussian_4096_*'))
        else:
            self.noise_files = sorted(glob.glob(os.path.join(noise_dir, "*.hdf5")))
            if len(self.noise_files) == 0:
                raise RuntimeError(f"No noise HDF5 files found in {noise_dir}")

            self.psd_H1_files = None
            self.psd_L1_files = None

        self.fs = sample_rate
        self.dt = 1 / self.fs
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.noise_prob = noise_prob
        self.noise_range = noise_range
        self.epoch = initial_epoch
        self.merger_out_prob = merger_out_prob
        self.fixed_epoch = validation_epoch
        self.edge_buffer = edge_buffer
        self.TRUNC = self.edge_buffer
        self.train = train
        self.plotsamples = False
        self.noise_handlers = [h5py.File(p, "r") for p in self.noise_files]

        # calculate params for boosting easier samples in earlier epochs
        self.p_higher_init = p_higher_init
        self.p_higher_fin  = p_higher_fin
        self.snr_range_high = tuple(snr_range_high)
        self.snr_range_low = tuple(snr_range_low)
        if np.isclose(self.p_higher_init, self.p_higher_fin):
            self.tau = np.inf
        else:
            self.tau = -10.0 / np.log(self.p_higher_fin / self.p_higher_init)


        # pre-compute band-pass coefficients
        self.band_low = band_low
        self.band_high = band_high
        self.bandpass_order = bandpass_order
        self.psd_floor = psd_floor
        self.psd_outband = psd_outband  # "infinite" PSD outside band
        
        nyq = 0.5 / self.dt
        self._butter = butter(
            self.bandpass_order,
            [self.band_low / nyq, self.band_high / nyq],
            btype='band',
        )
        self._psd_cache = {}
        



    def __len__(self):
        return len(self.indices)

    def increment_epoch(self):
        self.epoch += 1
        
    def __getitem__(self, index):
        file_idx, sample_idx = self.indices[index]
    
        if self.plotsamples:
            if self.noise_is_whitened:
                raise NotImplementedError("plot_samples is not supported for noise_is_whitened=True.")
            else:
                X, y, snr, wL_clean, wH_clean = self.__data_generation(file_idx, sample_idx, plot_samples=True)
            snr_arr = np.array([snr], dtype=np.float32)
            return (
                torch.from_numpy(X),
                torch.from_numpy(y),
                torch.from_numpy(snr_arr),
                torch.from_numpy(wL_clean.copy()),
                torch.from_numpy(wH_clean.copy()),
            )
        else:
            if self.noise_is_whitened:
                X, y = self.__data_generation_old(file_idx, sample_idx)
            else:
                X, y = self.__data_generation(file_idx, sample_idx)
            return (
                torch.from_numpy(X),
                torch.from_numpy(y),
            )
            
            
    def _make_band_limited_psd(self, freqs, psd_arr):
        """
        Return a psd that:
          - interpolates psd_arr defined on freqs
          - applies an absolute floor self.psd_floor
          - blows up PSD outside [band_low, band_high] to self.psd_outband
        """
        freqs = np.asarray(freqs)
        psd_arr = np.asarray(psd_arr)

        base_interp = scipy.interpolate.interp1d(
            freqs,
            psd_arr,
            bounds_error=False,
            fill_value=(psd_arr[0], psd_arr[-1]),
        )

        fmin = self.band_low
        fmax = self.band_high
        floor = self.psd_floor
        outval = self.psd_outband

        def psd_fun(f):
            f = np.asarray(f)
            psd = base_interp(f)
            psd = np.maximum(psd, floor)

            mask_out = (f < fmin) | (f > fmax)
            if mask_out.any():
                psd = np.where(mask_out, outval, psd)

            return psd

        return psd_fun
        
        
    def _window_ok_for_injection(self, segL, segH, k_sigma=5.0):
        """
        Quick local QC for a 1-second window used as a *positive* example.

        Returns True if both detectors look 'reasonable':
          - finite std
          - max|x| not larger than k_sigma * std

        We don't mind glitches in noise-only segments, but we want to avoid
        huge spikes in the same 1 s window that contains the merger (so the model doesn't learn glitch=merger).
        """
        for seg in (segL, segH):
            seg = np.asarray(seg)
            std = np.std(seg)
            if not np.isfinite(std) or std <= 0:
                return False

            max_abs = np.max(np.abs(seg))
            if max_abs > k_sigma * std:
                return False

        return True


            
    def _get_psd_interps_from_noisefile(self, idx_noise):
        """
        Return (psd_L1_interp, psd_H1_interp) for a given noise file index.

        Expects datasets 'psd_L1', 'psd_H1' and 'freqs' in the noise HDF5 file.
        """
        keyL = f"L1_{idx_noise}"
        keyH = f"H1_{idx_noise}"

        # return cached if available
        if keyL in self._psd_cache and keyH in self._psd_cache:
            return self._psd_cache[keyL], self._psd_cache[keyH]

        nf = self.noise_handlers[idx_noise]

        if ("psd_L1" not in nf) or ("psd_H1" not in nf):
            raise KeyError(
                f"Noise file {self.noise_files[idx_noise]} is missing "
                "'psd_L1' or 'psd_H1' datasets."
            )

        psd_L_arr = np.asarray(nf["psd_L1"][()])
        psd_H_arr = np.asarray(nf["psd_H1"][()])

        if "freqs" not in nf:
            raise KeyError(
                f"Noise file {self.noise_files[idx_noise]} has PSDs but no 'freqs' dataset; "
                "cannot build frequency-interpolated PSD."
            )

        freqs = np.asarray(nf["freqs"][()])

        # squeeze away potential singleton dims
        psd_L_arr = psd_L_arr.squeeze()
        psd_H_arr = psd_H_arr.squeeze()
        freqs     = freqs.squeeze()

        psd_L = self._make_band_limited_psd(freqs, psd_L_arr)
        psd_H = self._make_band_limited_psd(freqs, psd_H_arr)


        self._psd_cache[keyL] = psd_L
        self._psd_cache[keyH] = psd_H

        return psd_L, psd_H


    def __data_generation_old(self, file_idx, sample_idx):
        """ [LEGACY]  
        Requires pre-whitened noise chunks when used with this DataGenerator.
        Requires specifying a noise_snr_schedule or noise_range.
    
        Generates a single training sample (signal+noise or noise-only).
    
        For signal+noise samples:
        - Selects sample from waveform files and finds merger point.
        - Whitens signal using PSD.
        - Optionally shifts merger out of segment (`merger_out_prob`).
        - Mixes signal with whitened real LIGO noise at random SNR.
        - Creates target labels (1 near merger, 0 elsewhere).
    
        For noise-only samples:
        - Returns whitened LIGO noise, normalized.
        - Target is all zeros.
    
        Args:
            file_idx (int): Index of HDF5 file.
            sample_idx (int): Sample index within file.
    
        Returns:
            X (np.ndarray): Input [segment_length, n_channels] — L1 and H1.
            y (np.ndarray): Target [segment_length, 1] — 1 near merger (signal), 0 otherwise.
            
        not compatible with plot_sample_waveforms=True
        """

        X = np.zeros((self.segment_length, self.n_channels), dtype=np.float32)
        y = np.zeros((self.segment_length, 1), dtype=np.float32)

        f = self.file_handlers[file_idx]
        grp = f["data"]
        
        raw_H1 = grp["H1_wave"][sample_idx]
        raw_L1 = grp["L1_wave"][sample_idx]
        signal_len = raw_H1.shape[-1]

        #process signal+noise sample
        if np.random.random_sample() > self.noise_prob:
            merger_L1 = np.argmax(np.abs(raw_L1))
            merger_H1 = np.argmax(np.abs(raw_H1))
            shared_merger = (merger_L1 + merger_H1) // 2

            if self.gaussian:
                psd_L1 = pickle.load(open(self.psd_L1_files, 'rb'), encoding="bytes")
                psd_H1 = pickle.load(open(self.psd_H1_files, 'rb'), encoding="bytes")
                file_idx_noise = np.random.randint(len(self.noise_files))
                whitened_noise_strain_fn = self.noise_files[file_idx_noise]
            else:
                # NOTE: must load whitened noise to work with this dataloader
                file_idx_noise = np.random.randint(len(self.noise_files))
                whitened_noise_strain_fn = self.noise_files[file_idx_noise]
            
                nf = self.noise_handlers[file_idx_noise]  # already opened in __init__
                freqs = np.asarray(nf["freqs"][()]).squeeze()
                psd_L_arr = np.asarray(nf["psd_L1"][()]).squeeze()
                psd_H_arr = np.asarray(nf["psd_H1"][()]).squeeze()
            
                psd_L1 = scipy.interpolate.interp1d(
                    freqs, psd_L_arr, bounds_error=False, fill_value=(psd_L_arr[0], psd_L_arr[-1])
                )
                psd_H1 = scipy.interpolate.interp1d(
                    freqs, psd_H_arr, bounds_error=False, fill_value=(psd_H_arr[0], psd_H_arr[-1])
                )

            #whiten signal
            strain_whiten_L, strain_whiten_H = whiten.whiten_signal(
                raw_L1, raw_H1, self.dt, psd_L1, psd_H1
            )


            # zero out edges to remove whitening artifacts
            truncation = self.TRUNC
            strain_whiten_L[:truncation] = 0
            strain_whiten_L[-truncation:] = 0
            strain_whiten_H[:truncation] = 0
            strain_whiten_H[-truncation:] = 0

            #slice shorter segment from signal 
            #with probablity merger_out_prob, move merger outside of the segment to the right
            if np.random.rand() < self.merger_out_prob:
                post_merger_offset = np.random.randint(0, 3*self.segment_length // 4 + 1)
                start_idx = max(0, shared_merger - self.segment_length - post_merger_offset)
            else:
                start_idx = max(0, shared_merger - self.segment_length + np.random.randint(self.segment_length // 4, self.segment_length // 2))
            end_idx = start_idx + self.segment_length

            full_target = np.zeros_like(raw_L1)
            label_start = max(0, shared_merger - self.label_width)
            full_target[label_start:shared_merger + 1] = 1

            def slice_with_padding(arr, start_idx, end_idx, length):
                seg = np.zeros(length)
                valid_start = max(0, start_idx)
                valid_end = min(len(arr), end_idx)
                insert_start = max(0, -start_idx)
                insert_end = insert_start + (valid_end - valid_start)
                seg[insert_start:insert_end] = arr[valid_start:valid_end]
                return seg

            strain_whiten_L = slice_with_padding(strain_whiten_L, start_idx, end_idx, self.segment_length)
            strain_whiten_H = slice_with_padding(strain_whiten_H, start_idx, end_idx, self.segment_length)
            target_segment = slice_with_padding(full_target, start_idx, end_idx, self.segment_length)

            # consistent SNR for validation 
            current_epoch = self.fixed_epoch if self.fixed_epoch is not None else self.epoch
            if self.noise_range is not None:
                low_snr, high_snr = self.noise_range
            else:
                low_snr, high_snr = low_max_snr(current_epoch)

            noise_range = (low_snr, high_snr)
            
            if not hasattr(self, "_logged_snr_this_epoch"):
                label = "VAL" if self.fixed_epoch is not None else "TRAIN"
                epoch_val = self.fixed_epoch if self.fixed_epoch is not None else self.epoch
                print(f"[{label}][Epoch {epoch_val}] SNR range: {noise_range}")
                self._logged_snr_this_epoch = True


            #get noise chunk
            ligo_noise_L, ligo_noise_H = whiten.get_whitened_ligo_noise_chunk(
                strain_whiten_L, whitened_noise_strain_fn, gaussian=self.gaussian
            )
            
            #add signal and noise (rescale to match target_std)
            mixed_L, mixed_H = whiten.mix_signal_and_noise(
                strain_whiten_L, strain_whiten_H, ligo_noise_L, ligo_noise_H, noise_range
            )

            X[:, 0] = mixed_L
            X[:, 1] = mixed_H
            y[:, 0] = target_segment

        #process noise oly samples
        else:
            file_idx_noise = np.random.randint(len(self.noise_files))
            whitened_noise_strain_fn = self.noise_files[file_idx_noise]
            ligo_noise_L, ligo_noise_H = whiten.get_whitened_ligo_noise_chunk(
                np.zeros(self.segment_length), whitened_noise_strain_fn, gaussian=self.gaussian
            )

            X[:, 0] = ligo_noise_L / np.std(ligo_noise_L)
            X[:, 1] = ligo_noise_H / np.std(ligo_noise_H)
            y[:, 0] = 0

        return X, y

    def __data_generation(self, file_idx, sample_idx, plot_samples=False, force_inject=None, remaining_retries=5):
        

        """
        Generate a single training sample (signal+noise or noise-only).
    
        Requires *raw, unwhitened* LIGO noise chunks — whitening happens inside this method.
    
        For signal+noise samples:
        - Selects sample from waveform files and finds merger point.
        - Optionally rescales signal to target SNR (boost range 20–40) with decaying boost probability.
        - Injects signal at random position in noise.
        - Applies whitening and band-pass filtering.
        - Picks window near injected merger.
        - Sets target labels (1 near merger, 0 elsewhere).
    
        For noise-only samples:
        - Returns pure noise window (whitened + band-passed).
        - Target is all zeros.
    
        Args:
            file_idx (int): Index of HDF5 file.
            sample_idx (int): Sample index within file.
            plot_samples (bool): If True, returns extra info for visualization.
    
        Returns:
            If plot_samples=False:
                X (np.ndarray): Input [segment_length, n_channels] — L1 and H1.
                y (np.ndarray): Target [segment_length, 1].
    
            If plot_samples=True:
                X, y, snr, wL_clean, wH_clean:
                    - snr: Matched-filter SNR of injected signal (or 0 for noise-only).
                    - wL_clean / wH_clean: Clean whitened signal-only window (for plotting).
        """

        X = np.zeros((self.segment_length, self.n_channels), dtype=np.float32)
        y = np.zeros((self.segment_length, 1), dtype=np.float32)
    
        if force_inject is None:
            inject_signal = np.random.rand() >= self.noise_prob
        else:
            inject_signal = bool(force_inject)

    
        # ── waveform ──────────────────────────────────────────────
        f = self.file_handlers[file_idx]
        grp = f["data"]

        raw_H1 = grp["H1_wave"][sample_idx]    
        raw_L1 = grp["L1_wave"][sample_idx]   
        signal_len = raw_H1.shape[-1]

        merger_L1 = np.argmax(np.abs(raw_L1))
        merger_H1 = np.argmax(np.abs(raw_H1))
        shared_merg = (merger_L1 + merger_H1) // 2

        
            
        # ── noise file + PSD (cache) ─────────────────────────────
        
        #NOTE: must load raw unwhitened noise to work with this dataloader

        idx_noise = np.random.randint(len(self.noise_files))
        nf        = self.noise_handlers[idx_noise]

        if self.gaussian:
            if "gauss_L1" not in self._psd_cache:
                self._psd_cache["gauss_L1"] = whiten.load_interp_psd(self.psd_L1_files)
                self._psd_cache["gauss_H1"] = whiten.load_interp_psd(self.psd_H1_files)
            psd_L1 = self._psd_cache["gauss_L1"]
            psd_H1 = self._psd_cache["gauss_H1"]
        else:
            psd_L1, psd_H1 = self._get_psd_interps_from_noisefile(idx_noise)

        full_nL = nf['strain_L1']
        full_nH = nf['strain_H1']
    
        # ── slice noise once ─────────────────────────────────────
        buffer = self.TRUNC
        total_len = signal_len + 2 * buffer + self.segment_length
        
        noise_start = np.random.randint(0, len(full_nL) - total_len)
        nL = full_nL[noise_start : noise_start + total_len]
        nH = full_nH[noise_start : noise_start + total_len]
    
    
        '''# ── inject signal ────────────────────────────
        if inject_signal:
            
            if plot_samples:
                snr_L = whiten.matched_filter_snr(raw_L1, psd_L1, self.dt,self.psd_floor)
                snr_H = whiten.matched_filter_snr(raw_H1, psd_H1, self.dt,self.psd_floor)
                snr = np.sqrt(snr_L**2 + snr_H**2)


            self.p_higher = max(self.p_higher_fin, self.p_higher_init * np.exp(-self.epoch / self.tau))
            #if self.train==1:
            #    print('p_higher: '+str(self.p_higher)+' epoch:'+str(self.epoch))
            if np.random.rand() < self.p_higher:
                snr_L = whiten.matched_filter_snr(raw_L1, psd_L1, self.dt,self.psd_floor)
                snr_H = whiten.matched_filter_snr(raw_H1, psd_H1, self.dt,self.psd_floor)
                snr = np.sqrt(snr_L**2 + snr_H**2)

                #target_snr = np.random.uniform(15, 30) #boost snr range 15-30
                
                # Example SNR sampling weights
                r = np.random.rand()
                if r < 0.5:
                    # 50% of injections: just above threshold (weak / marginal)
                    target_snr = np.random.uniform(8, 15)
                elif r < 0.8:
                    # 30%: medium-loud
                    target_snr = np.random.uniform(15, 25)
                else:
                    # 20%: loud tail
                    target_snr = np.random.uniform(25, 45)



                scale = target_snr / (snr + 1e-6)
                raw_L1 *= scale
                raw_H1 *= scale
                snr = target_snr
                
            min_inject = buffer
            max_inject = total_len - signal_len - buffer
            inject_idx = np.random.randint(min_inject, max_inject)
            inj_end    = inject_idx + signal_len
            nL[inject_idx:inj_end] += raw_L1
            nH[inject_idx:inj_end] += raw_H1'''
            
        # ── inject signal ────────────────────────────
        if inject_signal:
            # compute raw SNR before any scaling
            snr_L = whiten.matched_filter_snr(raw_L1, psd_L1, self.dt, self.psd_floor)
            snr_H = whiten.matched_filter_snr(raw_H1, psd_H1, self.dt, self.psd_floor)
            snr0  = np.sqrt(snr_L**2 + snr_H**2)

            # update p_higher based on epoch (curriculum)
            if np.isfinite(self.tau):
                self.p_higher = max(
                    self.p_higher_fin,
                    self.p_higher_init * np.exp(-self.epoch / self.tau)
                )
            else:
                # no schedule: keep constant
                self.p_higher = self.p_higher_init

            # choose target SNR *inside* boosted regime:
            #   - with prob p_higher → "easy" high-SNR bin
            #   - otherwise         → "harder" lower-SNR bin
            u = np.random.rand()
            if u < self.p_higher:
                # easy / loud bin
                target_snr = np.random.uniform(*self.snr_range_high)
            else:
                # harder / quiet bin 
                target_snr = np.random.uniform(*self.snr_range_low)

            # rescale waveform to target SNR
            scale = target_snr / (snr0 + 1e-6)
            raw_L1 *= scale
            raw_H1 *= scale
            snr = target_snr  

            # inject scaled signal into noise
            min_inject = buffer
            max_inject = total_len - signal_len - buffer
            inject_idx = np.random.randint(min_inject, max_inject)
            inj_end    = inject_idx + signal_len
            nL[inject_idx:inj_end] += raw_L1
            nH[inject_idx:inj_end] += raw_H1

    
        # ── whiten + band-pass  ────────────────────
        b, a = self._butter
        #wL = whiten.whiten(filtfilt(b, a, nL), psd_L1, self.dt, self.psd_floor)[buffer:-buffer]
        #wH = whiten.whiten(filtfilt(b, a, nH), psd_H1, self.dt, self.psd_floor)[buffer:-buffer]
        wL = whiten.whiten(nL, psd_L1, self.dt)[buffer:-buffer] 
        wH = whiten.whiten(nH, psd_H1, self.dt)[buffer:-buffer]
        #wL = filtfilt(b, a, whiten.whiten(nL, psd_L1, self.dt))[buffer:-buffer] #filter high frequency artifacts from injection after whitening
        #wH = filtfilt(b, a, whiten.whiten(nH, psd_H1, self.dt))[buffer:-buffer]
        
        # ── pick window around merger ──────────────────────────────
        if inject_signal:
            bp_merger_idx  = inject_idx + shared_merg - buffer
            rel_merger_pos = np.random.randint(self.segment_length//2, self.segment_length)
            w0 = bp_merger_idx - rel_merger_pos
        else:
            max_start = len(wL) - self.segment_length
            w0 = np.random.randint(0, max_start + 1)
    
        w0 = max(0, min(w0, len(wL) - self.segment_length))
        w1 = w0 + self.segment_length
        
        segL = wL[w0:w1]
        segH = wH[w0:w1]
        
        # remove tiny DC offsets per window
        meansegL=np.mean(segL)
        meansegH=np.mean(segH)
        segL = segL - meansegL
        segH = segH - meansegH
        
        
        '''# --- local QC: avoid big glitches *inside the label region* for positives ---
        if inject_signal:
            rel_merger = bp_merger_idx - w0
            label_start = max(0, rel_merger - self.label_width)
            label_end   = max(0, min(self.segment_length, rel_merger))

            if label_end > label_start:  # sanity check
                roiL = segL[label_start:label_end]
                roiH = segH[label_start:label_end]

                # strict check first (e.g. k_sigma = 4)
                if not self._window_ok_for_injection(roiL, roiH, k_sigma=4.0):
                    if remaining_retries > 0:
                        # retry as a *signal* sample, but with one fewer retry
                        return self.__data_generation(
                            file_idx,
                            sample_idx,
                            plot_samples=plot_samples,
                            force_inject=True,
                            remaining_retries=remaining_retries - 1,
                        )
                    else:
                        # retries exhausted → try one last time with looser criterion
                        if not self._window_ok_for_injection(roiL, roiH, k_sigma=6.0):
                            # still too glitchy even with k_sigma=6 → fall back to noise-only
                            return self.__data_generation(
                                file_idx,
                                sample_idx,
                                plot_samples=plot_samples,
                                force_inject=False,   # noise-only
                                remaining_retries=0,  # and *no* further recursion
                            )
            
        # --- optional QC for noise-only windows: drop truly extreme glitches ---
        if not inject_signal:
            # check the whole window (or central part if you prefer)
            if not self._window_ok_for_injection(segL, segH, k_sigma=8.0):
                if remaining_retries > 0:
                    # resample a fresh noise-only window, fewer retries left
                    return self.__data_generation(
                        file_idx,
                        sample_idx,
                        plot_samples=plot_samples,
                        force_inject=False,
                        remaining_retries=remaining_retries - 1,
                    )
                # remaining_retries == 0 → accept even if it's ugly'''


        
        '''# normalize per window per detector
        stdsegL=np.std(segL)+1e-8
        stdsegH=np.std(segH)+1e-8
        stdsegshared = np.sqrt(0.5*(np.var(segL) + np.var(segH))) + 1e-8
        segL = segL / stdsegshared
        segH = segH / stdsegshared'''
        
        X[:, 0] = segL
        X[:, 1] = segH

    
        if inject_signal:
            rel_merger = bp_merger_idx - w0
            y[max(0, rel_merger - self.label_width): rel_merger, 0] = 1.0
    
        # ── plotting extras ──────────────────────────────────────
        if plot_samples:
            if inject_signal:
                
                sig_only_L = np.zeros_like(nL); sig_only_L[inject_idx:inj_end] = raw_L1
                sig_only_H = np.zeros_like(nH); sig_only_H[inject_idx:inj_end] = raw_H1
    
                #wL_clean = (filtfilt(b, a, whiten.whiten(sig_only_L, psd_L1, self.dt))[buffer:-buffer][w0:w1].copy()-meansegL)/stdsegshared
                #wH_clean = (filtfilt(b, a, whiten.whiten(sig_only_H, psd_H1, self.dt))[buffer:-buffer][w0:w1].copy()-meansegH)/stdsegshared
                wL_clean = whiten.whiten(sig_only_L, psd_L1, self.dt)[buffer:-buffer][w0:w1].copy()-meansegL
                wH_clean = whiten.whiten(sig_only_H, psd_H1, self.dt)[buffer:-buffer][w0:w1].copy()-meansegH
                #wL_clean = (whiten.whiten(filtfilt(b, a, sig_only_L), psd_L1, self.dt, self.psd_floor)[buffer:-buffer][w0:w1].copy()-meansegL)/stdsegshared
                #wH_clean = (whiten.whiten(filtfilt(b, a, sig_only_H), psd_H1, self.dt, self.psd_floor)[buffer:-buffer][w0:w1].copy()-meansegH)/stdsegshared
           
            
            else:
                snr = 0.0
                wL_clean = np.zeros_like(X[:, 0])
                wH_clean = np.zeros_like(X[:, 1])
    
            return X, y, snr, wL_clean, wH_clean
    
        return X, y


class WaveformDataModule(LightningDataModule):
    def __init__(self, noise_dir, data_dir, batch_size=32, dim=1024, n_channels=2,
                 shuffle=True, gaussian=False, noise_prob=0.7, noise_range=None,
                 num_workers=1, initial_epoch=0, segment_length=4096, edge_buffer=2048,
                 merger_out_prob=0.0, validation_epoch=10, p_higher_init=0.5, p_higher_fin=0.1,
                 snr_range_high=(10.0, 25.0), snr_range_low=(7.0, 15.0),
                 train_file="train.hdf", val_file="test.hdf",
                 noise_is_whitened=False,
                 sample_rate=4096, band_low=25.0, band_high=450.0,
                 bandpass_order=4, psd_floor=1e-48, psd_outband=1e40):
        super().__init__()
        self.batch_size = batch_size
        self.dim = dim
        self.gaussian = gaussian
        self.shuffle = shuffle
        self.data_dir = data_dir
        self.noise_dir = noise_dir
        self.n_channels = n_channels 
        self.noise_prob = noise_prob
        self.noise_range = noise_range
        self.num_workers = num_workers
        self.initial_epoch = initial_epoch
        self.segment_length = segment_length
        self.edge_buffer = edge_buffer
        self.merger_out_prob = merger_out_prob
        self.validation_epoch = validation_epoch
        self.p_higher_init=p_higher_init
        self.p_higher_fin=p_higher_fin
        self.snr_range_high = snr_range_high
        self.snr_range_low = snr_range_low
        self.train_file = train_file
        self.val_file = val_file
        self.noise_is_whitened = noise_is_whitened
        self.sample_rate = sample_rate
        self.band_low = band_low
        self.band_high = band_high
        self.bandpass_order = bandpass_order
        self.psd_floor = psd_floor
        self.psd_outband = psd_outband

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = GWDataset(
                self.noise_dir, self.data_dir, self.batch_size, dim=self.dim,
                n_channels=self.n_channels, shuffle=True, train=True, gaussian=self.gaussian, noise_prob=self.noise_prob,
                train_file=self.train_file, val_file=self.val_file,
                noise_is_whitened=self.noise_is_whitened,
                initial_epoch=self.initial_epoch, segment_length=self.segment_length, edge_buffer=self.edge_buffer,
                merger_out_prob=self.merger_out_prob,
                p_higher_init=self.p_higher_init, p_higher_fin=self.p_higher_fin,
                snr_range_high=self.snr_range_high, snr_range_low=self.snr_range_low,
                sample_rate=self.sample_rate, band_low=self.band_low, band_high=self.band_high,
                bandpass_order=self.bandpass_order, psd_floor=self.psd_floor, psd_outband=self.psd_outband
            )
    
            self.val_dataset = GWDataset(
                self.noise_dir, self.data_dir, self.batch_size, dim=self.dim,
                n_channels=self.n_channels, shuffle=False, train=False, gaussian=self.gaussian, noise_prob=self.noise_prob,
                train_file=self.train_file, val_file=self.val_file,
                noise_is_whitened=self.noise_is_whitened,
                initial_epoch=self.validation_epoch, segment_length=self.segment_length, edge_buffer=self.edge_buffer,
                merger_out_prob=self.merger_out_prob, validation_epoch=self.validation_epoch,
                p_higher_init=self.p_higher_init, p_higher_fin=self.p_higher_fin,
                snr_range_high=self.snr_range_high, snr_range_low=self.snr_range_low,
                sample_rate=self.sample_rate, band_low=self.band_low, band_high=self.band_high,
                bandpass_order=self.bandpass_order, psd_floor=self.psd_floor, psd_outband=self.psd_outband
            )

    def train_dataloader(self):
        
        use_distributed = (torch.distributed.is_available() and torch.distributed.is_initialized())

        sampler = DistributedSampler(self.train_dataset, shuffle=self.shuffle) if use_distributed else None

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=(self.shuffle if sampler is None else False),
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        
        use_distributed = (torch.distributed.is_available() and torch.distributed.is_initialized())

        sampler = DistributedSampler(self.val_dataset, shuffle=False) if use_distributed else None

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        


    def increment_epoch(self):
        self.epoch += 1
        if hasattr(self, "_logged_snr_this_epoch"):
            del self._logged_snr_this_epoch
