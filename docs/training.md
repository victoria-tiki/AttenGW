# Training

`train.py` trains the attention-based detector with PyTorch Lightning. It reads defaults from `configs/example.yaml` and allows command-line overrides.

## Run training

For a full run on a GPU cluster:

```bash
sbatch scripts/submit_train.slurm
```

For a small local smoke test:

```bash
python train.py --config configs/example.yaml --checkpoint_dir /tmp/attengw_train_test --batch_size 2 --num_workers 0
```

## Current recommended data flow

For new runs, use:

```yaml
shared:
  noise_is_whitened: false
```

This selects the current raw-noise dataloader path.

During training, the dataloader:

1. chooses either a waveform injection or a noise-only example,
2. selects a real detector-noise chunk,
3. injects the signal into noise when applicable,
4. uses PSD information from the noise file,
5. whitens and band-limits the sample,
6. returns an input tensor and binary target mask.

The model input has shape:

```text
[segment_length, 2]
```

corresponding to L1 and H1 channels.

The target has shape:

```text
[segment_length, 1]
```

and is 1 near the merger and 0 elsewhere. For noise-only examples, the target is all zeros.

## Main training parameters

```yaml
batch_size: 32
num_workers: 4
lr_init: 0.001
```

`batch_size` controls the number of examples per batch.

`num_workers` controls dataloader parallelism. Use `0` for local debugging if multiprocessing causes problems.

`lr_init` is the initial Adam learning rate.

## Data-generation parameters

```yaml
segment_length: 4096
dim: 1024
edge_buffer: 2048
noise_prob: 0.60
```

`segment_length` is the length of each training window in samples.

`dim` is the number of samples before merger labeled as signal.

`edge_buffer` is used to reduce whitening/filtering edge artifacts.

`noise_prob` is the probability of drawing a noise-only example. Increasing it can reduce false positives; decreasing it gives more signal examples.

## Curriculum parameters

```yaml
p_higher_init: 0.90
p_higher_fin: 0.25
```

The current dataloader can preferentially sample easier, higher-SNR examples early in training and reduce that probability over time.

Start with the defaults. Change these only after checking validation behavior and diagnostic plots.

## Diagnostics

At the start of training, the script computes basic validation-set diagnostics on rank 0, including raw strain amplitude statistics and matched-filter SNR statistics.

For the current raw-noise path, it also saves a preview figure:

```text
training_batch_preview.png
```

in `checkpoint_dir`.

This preview is useful for catching path/config mistakes before committing to a long run.

## Checkpoints and logs

Training writes checkpoints to:

```yaml
paths:
  checkpoint_dir: /path/to/checkpoints_dir
```

Checkpoints are saved with filenames like:

```text
model_attenGW-{epoch}-{val_loss}
```

The training script monitors validation loss and saves checkpoints during the run.

## Good practice

Tune parameters on validation data, not final test data.

Change one major parameter at a time.

Check diagnostic plots before interpreting training metrics.

Keep the downloaded noise, injection files, and inference data consistent in sampling rate and preprocessing assumptions.
