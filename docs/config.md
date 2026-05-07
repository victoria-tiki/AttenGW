# Configuration

The main user-facing configuration file is:

```bash
configs/example.yaml
```

It is used by both the downloader and training script. Command-line arguments can override many of the same values, but using the YAML file is the recommended starting point.

## `shared`

These settings should stay consistent between downloaded noise and training.

```yaml
shared:
  sample_rate: 4096
  noise_is_whitened: false
  band_low: 25.0
  band_high: 450.0
  bandpass_order: 4
  psd_floor: 1.0e-48
  psd_outband: 1.0e40
```

Use `noise_is_whitened: false` for new runs. This means the downloader saves raw detector strain and the training dataloader handles whitening and band-limiting internally.

Only use `noise_is_whitened: true` for legacy experiments that expect already whitened noise on disk.

`sample_rate` should match the injection files and downloaded noise files.

`band_low`, `band_high`, and `bandpass_order` control the frequency range used for downloader QC and training-time band-limiting. The default range is 25--450 Hz.

## `paths`

```yaml
paths:
  data_dir: /path/to/train_test_hdf_dir
  noise_dir: /path/to/noise_dir
  checkpoint_dir: /path/to/checkpoints_dir
```

`data_dir` should contain the injection files used for training and validation.

`noise_dir` is where the downloader writes HDF5 noise windows. It is also where training expects to find those files.

`checkpoint_dir` is where training writes checkpoints, logs, and diagnostic plots.

## `download`

Commonly changed options:

```yaml
download:
  gps_start: 1248652818
  gps_end: 1249862418
  window_len_s: 4096
  n_segments: 5
  mode: noise
  require_full_window: true
```

`mode` can be `noise`, `signal`, or `both`.

`event_pad_s` controls how much padding is excluded around known events when downloading noise-only windows.

The diagnostic plot options are:

```yaml
plot_timeline: true
plot_timeseries: true
plot_psd: true
target_plot_fs: 1024.0
```

The advanced QC options are:

```yaml
glitch_sigma: 5.0
glitch_max_frac: 0.01
max_std_ratio: 10.0
psd_seglen_s: 4.0
amp_thresh:
rms_thresh:
max_raw_std:
min_raw_std:
```

Leave optional thresholds blank to disable them.

## `training`

Commonly changed options:

```yaml
training:
  train_file: train.hdf
  val_file: val.hdf
  batch_size: 32
  num_workers: 4
  lr_init: 0.001
```

Data and label settings:

```yaml
dim: 1024
segment_length: 4096
edge_buffer: 2048
noise_prob: 0.60
```

`segment_length` is the number of samples fed to the model.

`dim` is the number of samples immediately before merger labeled as signal.

`edge_buffer` trims whitening/filtering edge artifacts.

`noise_prob` is the probability of drawing a noise-only training example. Raising it can make the model more conservative; lowering it gives the model more signal examples.

Curriculum settings:

```yaml
p_higher_init: 0.90
p_higher_fin: 0.25
```

These control how often the new dataloader samples higher-SNR injections early vs. later in training.
