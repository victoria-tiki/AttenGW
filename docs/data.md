# Data format and example data

This repository expects two kinds of data:

1. injection files containing clean detector-projected waveforms, used as signal examples;
2. noise files containing detector strain and PSDs, used by the dataloader to construct signal+noise and noise-only training examples.

A small example dataset is included under:

```text
example_data/
  signal/
  noise/
```

These files are intended only for smoke tests and basic end-to-end checks. They are not the full dataset used for paper-scale training.

The figure below shows example training samples constructed from these two data components. The first three rows contain injected gravitational-wave signals (orange) embedded in detector noise (blue), while the final row is a noise-only example. 

![Example training batch preview](../img/training_batch_preview8.png)

## Injection file format

The training and validation injection files should be HDF5 files with the following structure:

```text
/data/H1_wave   shape: (N, waveform_length)
/data/L1_wave   shape: (N, waveform_length)
```

where `N` is the number of injections in the file, and `waveform_length` is the number of time samples in each clean waveform.

The config file should point `paths.data_dir` to the directory containing these files, and `training.train_file` / `training.val_file` should name the specific train and validation files.

Example:

```yaml
paths:
  data_dir: example_data/signal

training:
  train_file: train_tiny.hdf
  val_file: val_tiny.hdf
```

The datagenerator reads the injection files as:

```text
<data_dir>/<train_file>
<data_dir>/<val_file>
```

and expects both files to contain `/data/H1_wave` and `/data/L1_wave`.

## Included tiny injection files

The example injection files in `example_data/signal/` contain synthetic detector-projected compact-binary waveforms. They are provided so that users can test the repository without downloading or generating a full injection dataset.

These files are for testing only. They are not intended to reproduce the training distribution or results of the paper.

The included example injections were generated with:

```text
waveform approximant: IMRPhenomXPHM
detectors: H1 and L1
sample rate: 4096 Hz
lower frequency cutoff: 20 Hz
distance range: 100–2500 Mpc
```

The sampled parameters used for the example set were:

```text
m1: power-law distribution p(m1) ∝ m1^-2.3 on [5, 50] solar masses
q = m2/m1: uniform on [0.1, 1], with m2 clipped to at least 5 solar masses
spin magnitudes: uniform on [0, 0.8]
spin directions: isotropic
inclination, polarization, RA, Dec: isotropic
distance: uniform in Euclidean volume
```

The waveforms were projected onto H1 and L1 using the detector response. The files may also contain a `/params` group with the sampled physical parameters, but the datagenerator only requires `/data/H1_wave` and `/data/L1_wave`.

## Noise file format

Noise files should be HDF5 files with the following structure:

```text
/strain_H1
/strain_L1
/psd_H1
/psd_L1
/freqs
```

The downloader script writes files in this format.

For the default dataloader path:

```yaml
shared:
  noise_is_whitened: false
```

the noise strain datasets should contain raw, unwhitened strain:

```text
/strain_H1
/strain_L1
```

The dataloader then uses the saved PSDs to whiten and band-limit the data internally.

For the legacy dataloader path:

```yaml
shared:
  noise_is_whitened: true
```

the noise strain datasets should already be whitened. The PSD datasets are still required because the dataloader uses them to whiten the injected clean signals.

## Included tiny noise file

The example noise file in `example_data/noise/` is a short raw-noise file intended for smoke testing. It is approximately 60 seconds long and contains:

```text
/strain_H1
/strain_L1
/psd_H1
/psd_L1
/freqs
```

The accompanying plots in the same folder are diagnostic outputs from the downloader, such as PSD and timeline plots. They are included only to show what the downloader produces and to make the example data easier to inspect.

## Running a small test with the example data

To run a small local or SLURM test using the included example files, edit the config file to point to the example folders:

```yaml
paths:
  data_dir: example_data/signal
  noise_dir: example_data/noise
  checkpoint_dir: checkpoints

training:
  train_file: train_tiny.hdf
  val_file: val_tiny.hdf

shared:
  noise_is_whitened: false
```

Then run training with:

```bash
python train.py --config configs/example.yaml
```

or submit through SLURM with:

```bash
sbatch scripts/submit.slurm
```

The example data is intentionally small. It is useful for checking that the dataloader, model, training loop, checkpoint saving, and plotting work end-to-end, but it should not be used to assess final model performance.
