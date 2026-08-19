# AttenGW for Gravitational Wave Classification

AttenGW is a gravitational-wave detection pipeline for classifying compact-binary merger signals in detector strain. The repository includes tools for downloading LIGO noise from GWOSC, training a detection model, and inspecting predictions from trained checkpoints. The results of this pipeline are further described in [arXiv:2512.12513](https://arxiv.org/abs/2512.12513).

---

## Basic usage

The basic workflow is:

1. install the requirements,
2. edit `configs/example.yaml`,
3. download real detector noise,
4. train on user-provided waveform injections,
5. run inference on downloader-produced HDF5 files using a trained checkpoint.
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

See [`docs/config.md`](docs/config.md) for the main configuration options. See [`docs/data.md`](docs/data.md) for more in-depth descriptions of expected data formats. Example datasets (both for injected signals and noise) that may be used to test AttenGW are provided in example_data. See `data.md` for a description of those datasets. By default, training uses `model_tcn_earlyfusion`, but the model can be changed through the YAML config (see `config.md` for how to change models, see [`docs/models.md`](docs/models.md) for an overview over all available models). 

---

## Download real detector noise

To download real detector noise (or, optionally, real signals to be used for inference) run:

```bash
python downloader.py --config configs/example.yaml
```

Optional SLURM usage:

```bash
sbatch scripts/submit_download.slurm
```

The downloader can save diagnostic timeline, time-series, and PSD plots depending on the options in `configs/example.yaml`.

See [`docs/config.md`](docs/config.md) for downloader configuration options to be changed through the config script.

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

See [`docs/config.md`](docs/config.md) for an overview of training parameters to be changed through the config script. See [`docs/training.md`](docs/training.md) for the training data flow and the main parameters worth changing. 

---

## Inference

After training, use the inference script to run a trained checkpoint on a downloader-produced HDF5 file and report human-readable trigger times.

First edit:

```bash
configs/inference_example.yaml
```
At minimum, set:
```
checkpoint:
  run_dir: /path/to/checkpoints/latest_run_id
  ckpt_path:

input:
  file: /path/to/noise_or_signal_file.hdf5

output:
  output_dir: /path/to/inference_outputs

```
`checkpoint.run_dir` should point to a training run folder containing `config.yaml` (will be saved automatically by the train script) and one or more model `.ckpt` files. If `checkpoint.ckpt_path` is left blank, the inference script will use the newest checkpoint in the run folder.

Run locally with:

```
python inference/infer.py --config configs/inference_example.yaml
```
Optional SLURM usage:

```
sbatch scripts/submit_infer.slurm
```

The script saves trigger summaries and optional diagnostic plots to the configured output directory. See [`docs/inference.md`](docs/inference.md) for notes on interpreting model scores and peak-finding settings.

---


## Repository map

```text
.github/                    GitHub metadata and repository ownership files
configs/example.yaml        Example configuration for downloading and training
configs/inference_example.yaml    Example configuration for inference
inference/infer.py                Basic inference script for finding triggers
inference/inference_utils.py      Shared inference utilities
docs/                       Extended documentation
example_data/signal/        Tiny example injection file
example_data/noise/         Tiny example noise file and downloader diagnostics
img/                        README and documentation images
models/                          Config-selectable model definitions
scripts/submit_download.slurm  SLURM script for downloading noise data
scripts/submit_train.slurm     SLURM script for launching training
scripts/submit_infer.slurm     SLURM script for running inference
downloader.py               Downloads GWOSC strain windows and PSDs
data_generator.py           Builds training windows from injections and noise
model.py                    Attention-based detector model
train.py                    PyTorch Lightning training script
requirements.txt            Python dependencies
```

## Data and checkpoints

Large training, validation, or noise files are not stored in the repository. Generate or download them locally and point the config paths to their location. Small example datasets are provided in `/example_data`. These are to be used for testing only (see [`docs/data.md`](docs/data.md)). Selected pretrained O3b checkpoints are provided for testing and inference in ```/checkpoints```.
