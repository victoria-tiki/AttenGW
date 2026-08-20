# AttenGW: Lightweight Neural Gravitational-Wave Detection in Real LIGO Noise

AttenGW is a gravitational-wave detection pipeline for developing and evaluating lightweight neural gravitational-wave detectors in real LIGO noise. The repository includes tools for downloading LIGO noise from GWOSC, constructing training examples from synthetic waveform injections and real detector noise, training a detection model, running inference, , and constructing candidate triggers, and evaluating robustness under detector-noise domain shift.

The project was originally developed around an HDCN cross-attention model. The current repository also includes several other architectures; the default configuration uses the lightweight early-fusion TCN selected in the accompanying development study.

The results of this pipeline are further described in [arXiv:2512.12513](https://arxiv.org/abs/2512.12513).

---

## Basic usage

The basic workflow is:

1. install the requirements,
2. edit `configs/example.yaml`,
3. download real detector noise,
4. train a selected model using user-provided waveform files and downloaded noise,
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

download:
  train_noise:
    gps_start: <TRAIN_START_GPS>
    gps_end: <TRAIN_END_GPS>

  test:
    gps_start: <TEST_START_GPS>
    gps_end: <TEST_END_GPS>

training:
  train_file: train.hdf
  val_file: val.hdf
```

Replace the GPS placeholders with non-overlapping training and test intervals. `data_dir` should contain the waveform injection files named by `training.train_file` and `training.val_file`. Each file is expected to contain:

```text
/data/H1_wave
/data/L1_wave
```

with shape:

```text
(N, waveform_length)
```

The waveforms are dynamically combined with real detector noise during training.

`noise_dir` is the root directory for downloader-produced data. The downloader writes files under:

```text
noise_dir/
├── train/
└── test/
    ├── noise/
    └── signal/
```

Each HDF5 file contains:

```text
/strain_H1
/strain_L1
/psd_H1
/psd_L1
/freqs
```

The files also store relevant metadata.

Small example files illustrating the expected signal and noise formats are provided under `example_data/`. The default model is `model_tcn_earlyfusion`; other models can be selected through `model.name` in the YAML configuration.

See [`docs/config.md`](docs/config.md) and [`docs/data.md`](docs/data.md) for detailed configuration and data-format documentation.


---

## Download real detector noise

To download training noise and, optionally, held-out noise or catalog-event windows, configure the corresponding entries under `download` in `configs/example.yaml`, then run:

```bash
python downloader.py --config configs/example.yaml
```

Optional SLURM usage:

```bash
sbatch scripts/submit_download.slurm
```

Files are written under `train/`, `test/noise/`, and `test/signal/` within the configured `noise_dir`. The downloader can also save diagnostic timeline, time-series, and PSD plots.

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
configs/              Download, training, and inference configurations
docs/                 Extended documentation
example_data/         Small example signal and noise files
models/               Config-selectable model implementations
inference/            Inference and trigger-construction scripts
scripts/              SLURM launchers for download, training, and inference
checkpoints/          Selected pretrained runs
downloader.py         Downloads GWOSC training and test data
data_generator.py     Builds training examples from waveforms and noise
train.py              PyTorch Lightning training entry point
requirements.txt      Python dependencies
```

## Data and checkpoints

Large signal or noise files are not stored in the repository. Generate or download them locally and point the config paths to their location. 

Small example datasets are provided in `example_data/`. These are to be used for testing and inspection (see [`docs/data.md`](docs/data.md)). 

Selected pretrained runs, including their saved configurations and checkpoints, are provided under `checkpoints/` for testing and inference.
