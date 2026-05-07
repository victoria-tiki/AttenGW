# Legacy mode

The default workflow for new runs should use raw detector noise:

```yaml
shared:
  noise_is_whitened: false
```

Legacy mode is selected with:

```yaml
shared:
  noise_is_whitened: true
```

Use legacy mode only if you are reproducing older experiments or using older noise files that were already whitened before training.

## What legacy mode changes

Setting `noise_is_whitened: true` does more than change the expected noise format. It switches the dataset to the older dataloader path in `data_generator.py`.

The main differences are:

| Behavior | Current raw-noise path | Legacy whitened-noise path |
|---|---|---|
| Noise on disk | Raw detector strain plus PSDs | Already whitened detector strain |
| Signal/noise mixing | Injects into raw noise, then whitens/band-limits the combined sample | Whitens the signal first, then mixes it with already-whitened noise |
| SNR meaning | Matched-filter-style SNR | Relative time-domain noise/signal scaling |
| Curriculum | Uses `p_higher_init` and `p_higher_fin` | Uses `noise_range` or the legacy `low_max_snr(epoch)` schedule |
| Band-limiting | Handled in the current dataloader | Assumed to be handled before or during old preprocessing |
| Diagnostics | Supports `plot_samples` preview | Training skips `plot_samples` for this path |

## Legacy SNR behavior

Legacy mode does not use the same SNR definition as the current raw-noise path.

In the current path, SNR is closer to a matched-filter SNR. In legacy mode, the code scales already-whitened noise relative to the whitened signal using either `noise_range` or the built-in epoch schedule.

Because of this, legacy SNR values should not be compared directly to matched-filter SNR values from the current path.

## When to use it

Use legacy mode if:

- you need to reproduce older experiments,
- your noise files are already whitened,
- you are comparing against checkpoints trained with the old pipeline.

For new runs, prefer:

```yaml
shared:
  noise_is_whitened: false
```

## Caveats

The current config defaults, downloader behavior, and main documentation are written for the raw-noise path.

`plot_samples` is skipped when `noise_is_whitened=True`, so the automatic training-batch preview is only available for the current path.

Because legacy mode assumes preprocessed noise, reproducibility depends on knowing how those whitened noise files were originally created. If that preprocessing is unclear, regenerate raw-noise files and use the current workflow.
