# Data and downloader

`downloader.py` downloads H1/L1 detector strain from GWOSC, applies optional quality-control checks, estimates PSDs, and saves HDF5 noise or signal windows for training.

For new runs, the recommended setup is:

```yaml
shared:
  noise_is_whitened: false
```

With this setting, the downloader saves raw detector strain and PSD information. Whitening is handled during training.

## Run the downloader

```bash
python downloader.py --config configs/example.yaml
```

Optional SLURM usage:

```bash
sbatch scripts/submit_download.slurm
```

The downloader reads defaults from the YAML config and then accepts command-line overrides.

## Output files

The downloader saves HDF5 files in `paths.noise_dir`. Training expects noise files containing:

```text
/strain_H1
/strain_L1
/psd_H1
/psd_L1
/freqs
```

These are the files consumed by the dataloader during training.

## Required injection files

Training also needs user-provided waveform injection files in `paths.data_dir`. The training and validation filenames are configured under:

```yaml
training:
  train_file: train.hdf
  val_file: val.hdf
```

Each injection file should contain:

```text
/data/H1_wave
/data/L1_wave
```

with shape:

```text
(N, waveform_length)
```

where `N` is the number of waveforms.

## Downloader modes

```yaml
mode: noise
```

Allowed values:

```text
noise
signal
both
```

For most training runs, use `noise` to download event-free detector noise and provide injection files separately.

## Window selection

```yaml
gps_start: 1248652818
gps_end: 1249862418
window_len_s: 4096
n_segments: 5
require_full_window: true
```

`gps_start` and `gps_end` define the search interval.

`window_len_s` is the duration of each downloaded window in seconds.

`n_segments` limits how many accepted windows are saved. Use `null` if you want as many as possible.

`require_full_window: true` skips partial windows.

## Event exclusion

```yaml
event_pad_s: 30.0
```

In noise mode, the downloader excludes windows around known events, padded by this amount.

## Quality-control settings

The downloader can reject windows using simple detector-strain checks. These checks are meant to avoid obviously bad data, not to fully certify science-grade data quality.

```yaml
glitch_sigma: 5.0
glitch_max_frac: 0.01
max_std_ratio: 10.0
amp_thresh:
rms_thresh:
max_raw_std:
min_raw_std:
```

`glitch_sigma` identifies short loud glitches using a robust MAD-based threshold.

`glitch_max_frac` rejects a window if too many samples are flagged.

`max_std_ratio` rejects windows where H1 and L1 have very different standard deviations.

Leave `amp_thresh`, `rms_thresh`, `max_raw_std`, and `min_raw_std` blank to disable those checks.

## Diagnostic plots

```yaml
plot_timeline: true
plot_timeseries: true
plot_psd: true
target_plot_fs: 1024.0
```

These options save quick diagnostic figures. They are useful for checking whether the chosen GPS range and QC settings are producing reasonable files.
