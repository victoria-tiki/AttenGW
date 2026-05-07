# AttenGW for Gravitational Wave Classification

AttenGW is an attention-based gravitational-wave detection pipeline for classifying compact-binary merger signals in detector strain. The repository includes tools for downloading LIGO noise from GWOSC, training the detection model, and inspecting predictions from trained checkpoints. The results of this model are further described in [arXiv:2512.12513](https://arxiv.org/abs/2512.12513).

---

## Basic usage

The basic workflow is:

1. install the requirements,
2. edit `configs/example.yaml`,
3. download real detector noise,
4. train on user-provided waveform injections,
5. use the inference notebook to inspect predictions from a trained checkpoint.

--- 

## Installation

Python 3.10 or newer is recommended. The code has been tested with Python 3.11.

```bash
python3.11 -m venv .venv_attengw
source .venv_attengw/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Tested with:

```text
Python 3.11.13
PyTorch 2.11.0+cu130
PyTorch Lightning 2.6.1
```

---

## Configure paths and parameters

Edit:

```bash
configs/example.yaml
```

At minimum, set:

```yaml
paths:
  data_dir: /path/to/train_test_hdf_dir
  noise_dir: /path/to/noise_dir
  checkpoint_dir: /path/to/checkpoints_dir
```

`data_dir` should contain the waveform injection files named by `training.train_file` and `training.val_file`. Each file is expected to contain:

```text
/data/H1_wave
/data/L1_wave
```

with shape:

```text
(N, waveform_length)
```

`noise_dir` is the output directory for downloaded detector noise. The downloader creates HDF5 files containing:

```text
/strain_H1
/strain_L1
/psd_H1
/psd_L1
/freqs
```

See [`docs/config.md`](docs/config.md) for the main configuration options.

---

## Download real detector noise

For the recommended raw-noise workflow, keep:

```yaml
shared:
  noise_is_whitened: false
```

Then run:

```bash
python downloader.py --config configs/example.yaml
```

Optional SLURM usage:

```bash
sbatch scripts/submit_download.slurm
```

The downloader can save diagnostic timeline, time-series, and PSD plots depending on the options in `configs/example.yaml`.

See [`docs/data.md`](docs/data.md) for downloader behavior, output format, and QC options.

---

## Train

Training is much faster on a GPU. For full training runs, use:

```bash
sbatch scripts/submit_train.slurm
```

For a small local smoke test:

```bash
python train.py --config configs/example.yaml --checkpoint_dir /tmp/attengw_train_test --batch_size 2 --num_workers 0
```

The training script loads defaults from the YAML config, but command-line arguments can override them.

See [`docs/training.md`](docs/training.md) for the training data flow and the main parameters worth changing.

---

## Inference

After training, use the inference notebook to load a checkpoint and visualize predictions on selected examples.

See [`docs/inference.md`](docs/inference.md) for notes on interpreting model scores and peak-finding settings.

---

## Legacy behavior

The old `noise_is_whitened=True` path is retained for reproducing older experiments that used pre-whitened noise. New runs should normally use the default raw-noise workflow.

See [`docs/legacy.md`](docs/legacy.md) before using `noise_is_whitened=True`.

---

## Repository map

```text
configs/example.yaml        Example config used by downloader and training
downloader.py               Downloads GWOSC strain windows and PSDs
data_generator.py           Builds training windows from injections and noise
model.py                    Attention-based detector model
train.py                    PyTorch Lightning training script
requirements.txt            Python dependencies
scripts/submit_download.slurm
scripts/submit_train.slurm
docs/                       Extended documentation
```

## Data and checkpoints

Large training, validation, noise, and checkpoint files are not stored in the repository. Generate or download them locally and point the config paths to their location.
