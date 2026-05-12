import glob
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from astropy.time import Time
from scipy import signal
from torch.utils.data import DataLoader, Dataset

# Allow running as: python inference/infer.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import full_module
from data_generator import GWDataset, whiten


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_checkpoint(run_dir: str, ckpt_path: Optional[str] = None) -> str:
    """Return explicit checkpoint path or newest .ckpt file in run_dir."""
    if ckpt_path:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    candidates = sorted(
        glob.glob(os.path.join(run_dir, "*.ckpt")),
        key=os.path.getmtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No .ckpt files found in {run_dir}")

    return candidates[-1]


def load_training_config(run_dir: str) -> Optional[Dict[str, Any]]:
    """Load config.yaml saved in a training run folder, if present."""
    path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(path):
        print(f"WARNING: no training config found at {path}")
        return None
    return load_yaml(path)


def choose_device(device_setting: str = "auto") -> torch.device:
    if device_setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_setting)


def load_model_from_checkpoint(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load full_module from a Lightning or plain PyTorch checkpoint.

    Training checkpoints from Lightning usually store weights under "state_dict"
    with a "model." prefix, because LightningModel contains self.model = full_module().
    """
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model = full_module()
    model_state = model.state_dict()

    cleaned = {}
    for key, value in state_dict.items():
        name = key
        for prefix in ["model.", "net.", "module."]:
            if name.startswith(prefix):
                name = name[len(prefix):]

        if name in model_state and model_state[name].shape == value.shape:
            cleaned[name] = value

    missing_fraction = 1.0 - (len(cleaned) / max(1, len(model_state)))
    if missing_fraction > 0.25:
        print(
            f"WARNING: loaded only {len(cleaned)}/{len(model_state)} model tensors. "
            "This may indicate a checkpoint/model mismatch."
        )
    else:
        print(f"Loaded {len(cleaned)}/{len(model_state)} model tensors.")

    model_state.update(cleaned)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model


def read_downloader_hdf5(path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Read a downloader-produced HDF5 file.

    Expected datasets:
      strain_H1, strain_L1, psd_H1, psd_L1, freqs

    PSD datasets are required for raw files. Pre-whitened files can still include
    PSDs for provenance and sanity checks.
    """
    with h5py.File(path, "r") as f:
        data = {
            "strain_H1": np.asarray(f["strain_H1"][:], dtype=np.float64),
            "strain_L1": np.asarray(f["strain_L1"][:], dtype=np.float64),
        }

        for key in ["psd_H1", "psd_L1", "freqs"]:
            if key in f:
                data[key] = np.asarray(f[key][:], dtype=np.float64)

        attrs = {key: value for key, value in f.attrs.items()}

    return data, attrs


def _attr_bool(attrs: Dict[str, Any], key: str, default: bool = False) -> bool:
    if key not in attrs:
        return default
    return bool(attrs[key])


def _warn_or_raise(message: str, strict: bool) -> None:
    if strict:
        raise ValueError(message)
    print(f"WARNING: {message}")


def check_metadata_compatibility(
    training_config: Optional[Dict[str, Any]],
    file_attrs: Dict[str, Any],
    checks: Dict[str, Any],
) -> None:
    """
    Compare downloader file attrs to the training config saved with the checkpoint.

    Mismatches warn by default rather than failing.
    """
    if training_config is None:
        return

    strict = bool(checks.get("strict", False))
    shared = training_config.get("shared", {})

    if checks.get("check_sample_rate", True):
        train_fs = shared.get("sample_rate")
        file_fs = file_attrs.get("sample_rate")
        if train_fs is not None and file_fs is not None and int(train_fs) != int(file_fs):
            _warn_or_raise(
                f"sample_rate mismatch: training config has {train_fs}, "
                f"input file has {file_fs}.",
                strict,
            )

    if checks.get("check_whitened_state", True):
        train_whitened = bool(shared.get("noise_is_whitened", False))
        file_whitened = _attr_bool(file_attrs, "whiten", default=False)
        if train_whitened != file_whitened:
            _warn_or_raise(
                f"whitening-state mismatch: training config has "
                f"noise_is_whitened={train_whitened}, input file has "
                f"whiten={file_whitened}. Continuing may be valid for testing, "
                f"but the model may see differently preprocessed data.",
                strict,
            )

    if checks.get("check_bandpass", True):
        for key in ["band_low", "band_high", "bandpass_order"]:
            train_val = shared.get(key)
            file_val = file_attrs.get(key)
            if train_val is None or file_val is None:
                continue

            if float(train_val) != float(file_val):
                _warn_or_raise(
                    f"{key} mismatch: training config has {train_val}, "
                    f"input file has {file_val}.",
                    strict,
                )


def make_psd_interps_from_arrays(
    freqs: np.ndarray,
    psd_L1: np.ndarray,
    psd_H1: np.ndarray,
    sample_rate: float,
    band_low: float,
    band_high: float,
    psd_floor: float,
    psd_outband: float,
):
    """
    Build band-limited PSD interpolants using the same helper as GWDataset.

    This mirrors the inference utility pattern you already had, but uses values
    from config/metadata rather than hard-coded constants.
    """
    psd_self = object.__new__(GWDataset)
    psd_self.dt = 1.0 / float(sample_rate)
    psd_self.psd_floor = float(psd_floor)
    psd_self.psd_outband = float(psd_outband)
    psd_self.band_low = float(band_low)
    psd_self.band_high = float(band_high)

    psd_L = GWDataset._make_band_limited_psd(psd_self, freqs, psd_L1.squeeze())
    psd_H = GWDataset._make_band_limited_psd(psd_self, freqs, psd_H1.squeeze())
    return psd_L, psd_H


def prepare_strain_for_inference(
    data: Dict[str, np.ndarray],
    attrs: Dict[str, Any],
    training_config: Optional[Dict[str, Any]],
    inference_config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Return prepared L1/H1 arrays and bookkeeping metadata.

    Raw downloader file:
      whiten strain_L1/H1 using psd_L1/H1 and freqs, then trim edge_buffer.

    Pre-whitened downloader file:
      use strain_L1/H1 as-is. No additional whitening.
    """
    inf = inference_config["inference"]
    shared = (training_config or {}).get("shared", {})

    sample_rate = float(attrs.get("sample_rate", shared.get("sample_rate", 4096)))
    band_low = float(attrs.get("band_low", shared.get("band_low", 25.0)))
    band_high = float(attrs.get("band_high", shared.get("band_high", 450.0)))
    psd_floor = float(shared.get("psd_floor", 1e-48))
    psd_outband = float(shared.get("psd_outband", 1e40))
    edge_buffer = int(inf.get("edge_buffer", 2048))

    file_whitened = _attr_bool(attrs, "whiten", default=False)

    strain_L1 = np.asarray(data["strain_L1"], dtype=np.float64)
    strain_H1 = np.asarray(data["strain_H1"], dtype=np.float64)

    trim_samples = 0

    if file_whitened:
        print("Input file is marked whiten=True; using strain_H1/L1 as pre-whitened.")
        prep_L1 = strain_L1
        prep_H1 = strain_H1
    else:
        print("Input file is marked whiten=False; whitening using PSDs saved in the HDF5 file.")

        required = ["psd_L1", "psd_H1", "freqs"]
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(
                "Raw input file requires PSD datasets for inference. Missing: "
                + ", ".join(missing)
            )

        psd_L, psd_H = make_psd_interps_from_arrays(
            freqs=data["freqs"],
            psd_L1=data["psd_L1"],
            psd_H1=data["psd_H1"],
            sample_rate=sample_rate,
            band_low=band_low,
            band_high=band_high,
            psd_floor=psd_floor,
            psd_outband=psd_outband,
        )

        dt = 1.0 / sample_rate
        prep_L1 = whiten.whiten(strain_L1, psd_L, dt, floor=psd_floor)
        prep_H1 = whiten.whiten(strain_H1, psd_H, dt, floor=psd_floor)

        if edge_buffer > 0 and len(prep_L1) > 2 * edge_buffer:
            prep_L1 = prep_L1[edge_buffer:-edge_buffer]
            prep_H1 = prep_H1[edge_buffer:-edge_buffer]
            trim_samples = edge_buffer

    n = min(len(prep_L1), len(prep_H1))
    prep_L1 = prep_L1[:n]
    prep_H1 = prep_H1[:n]

    # Match the simple inference utilities: remove DC and normalize each detector.
    # This makes the scale manageable for both raw-whitened and pre-whitened inputs.
    prep_L1 = prep_L1 - np.mean(prep_L1)
    prep_H1 = prep_H1 - np.mean(prep_H1)

    std_L1 = np.std(prep_L1)
    std_H1 = np.std(prep_H1)
    if std_L1 > 0:
        prep_L1 = prep_L1 / std_L1
    if std_H1 > 0:
        prep_H1 = prep_H1 / std_H1

    info = {
        "sample_rate": sample_rate,
        "file_whitened": file_whitened,
        "trim_samples": trim_samples,
        "effective_gps_start": float(attrs.get("gps_start", 0.0)) + trim_samples / sample_rate,
        "gps_end": float(attrs.get("gps_end", np.nan)),
    }
    return prep_L1.astype(np.float32), prep_H1.astype(np.float32), info


class WindowDataset(Dataset):
    """Create fixed-length windows from two prepared detector streams."""

    def __init__(
        self,
        strain_L1: np.ndarray,
        strain_H1: np.ndarray,
        segment_length: int,
        stride: int,
        offset: int,
    ):
        self.strain_L1 = strain_L1
        self.strain_H1 = strain_H1
        self.segment_length = int(segment_length)
        self.stride = int(stride)
        self.offset = int(offset)

        n = min(len(strain_L1), len(strain_H1))
        last_start = n - self.segment_length
        if self.offset > last_start:
            self.starts = np.array([], dtype=int)
        else:
            self.starts = np.arange(self.offset, last_start + 1, self.stride, dtype=int)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        start = int(self.starts[idx])
        end = start + self.segment_length

        x = np.stack(
            [self.strain_L1[start:end], self.strain_H1[start:end]],
            axis=-1,
        ).astype(np.float32)

        return torch.from_numpy(x), start


def predict_offset_stream(
    model: torch.nn.Module,
    strain_L1: np.ndarray,
    strain_H1: np.ndarray,
    segment_length: int,
    stride: int,
    offset: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict one offset stream.

    Returns
    -------
    preds : np.ndarray
        Concatenated predictions for the offset stream.
    starts : np.ndarray
        Window start sample for each segment in this stream.
    """
    dataset = WindowDataset(strain_L1, strain_H1, segment_length, stride, offset)
    if len(dataset) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=int)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    starts = []

    model.eval()
    with torch.no_grad():
        for x, start in loader:
            x = x.to(device)
            y = model(x)
            if isinstance(y, (tuple, list)):
                y = y[0]
            y = y.detach().cpu().numpy()

            if y.ndim == 3:
                y = y.reshape(y.shape[0], -1)
            elif y.ndim == 1:
                y = y[None, :]

            preds.append(y)
            starts.extend(start.numpy().tolist())

    return np.concatenate(preds, axis=0).reshape(-1), np.asarray(starts, dtype=int)


def find_trigger_regions(
    preds: np.ndarray,
    threshold: float,
    min_width_samples: int,
    sample_rate: float,
    mean_margin: float = 0.05,
    mean_cap: float = 0.95,
) -> List[Dict[str, Any]]:
    """
    Find high-confidence trigger regions in one prediction stream.

    This follows the older inference scripts: scipy find_peaks, a width cut,
    and a dynamic mean-above-threshold cut.
    """
    if len(preds) == 0:
        return []

    dynamic_mean = max(0.0, min(float(mean_cap), float(threshold) - float(mean_margin)))

    peaks, props = signal.find_peaks(
        preds,
        height=float(threshold),
        width=int(min_width_samples),
        distance=int(sample_rate),
    )

    left_ips = props.get("left_ips", [])
    right_ips = props.get("right_ips", [])

    regions = []
    for peak, left, right in zip(peaks, left_ips, right_ips):
        left_i = int(max(0, np.floor(left)))
        right_i = int(min(len(preds) - 1, np.ceil(right)))
        sliced = preds[left_i:right_i + 1]

        if sliced.size == 0:
            continue

        if np.mean(sliced > dynamic_mean) <= 0.5:
            continue

        regions.append(
            {
                "peak_index": int(peak),
                "left_index": left_i,
                "right_index": right_i,
                "peak_score": float(preds[int(peak)]),
                "mean_cut": float(dynamic_mean),
            }
        )

    return regions


def merge_triggers(triggers: List[Dict[str, Any]], merge_tolerance_s: float) -> List[Dict[str, Any]]:
    """Merge duplicate triggers from different offset streams."""
    if not triggers:
        return []

    triggers = sorted(triggers, key=lambda x: x["peak_time_s"])
    merged = [triggers[0]]

    for trig in triggers[1:]:
        last = merged[-1]
        if trig["peak_time_s"] - last["peak_time_s"] <= merge_tolerance_s:
            # Keep the higher-score duplicate.
            if trig["peak_score"] > last["peak_score"]:
                merged[-1] = trig
        else:
            merged.append(trig)

    return merged


def triggers_for_threshold_and_width(
    model: torch.nn.Module,
    strain_L1: np.ndarray,
    strain_H1: np.ndarray,
    sample_rate: float,
    effective_gps_start: float,
    segment_length: int,
    stride: int,
    offsets: Iterable[int],
    batch_size: int,
    threshold: float,
    min_width_samples: int,
    mean_margin: float,
    mean_cap: float,
    merge_tolerance_s: float,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], Dict[int, np.ndarray]]:
    """
    Run model over all offsets and return merged human-readable trigger dicts.
    """
    all_triggers = []
    pred_by_offset = {}

    for offset in offsets:
        preds, _ = predict_offset_stream(
            model=model,
            strain_L1=strain_L1,
            strain_H1=strain_H1,
            segment_length=segment_length,
            stride=stride,
            offset=int(offset),
            batch_size=batch_size,
            device=device,
        )

        pred_by_offset[int(offset)] = preds

        regions = find_trigger_regions(
            preds=preds,
            threshold=threshold,
            min_width_samples=min_width_samples,
            sample_rate=sample_rate,
            mean_margin=mean_margin,
            mean_cap=mean_cap,
        )

        for region in regions:
            # Within an offset stream, preds are concatenated contiguous windows.
            peak_sample = int(region["peak_index"] + int(offset))
            left_sample = int(region["left_index"] + int(offset))
            right_sample = int(region["right_index"] + int(offset))

            peak_time_s = peak_sample / sample_rate
            left_time_s = left_sample / sample_rate
            right_time_s = right_sample / sample_rate
            peak_gps = effective_gps_start + peak_time_s

            region.update(
                {
                    "offset": int(offset),
                    "peak_sample": peak_sample,
                    "left_sample": left_sample,
                    "right_sample": right_sample,
                    "peak_time_s": float(peak_time_s),
                    "left_time_s": float(left_time_s),
                    "right_time_s": float(right_time_s),
                    "peak_gps": float(peak_gps),
                    "peak_utc": Time(peak_gps, format="gps").to_datetime().isoformat(),
                }
            )
            all_triggers.append(region)

    merged = merge_triggers(all_triggers, merge_tolerance_s=merge_tolerance_s)
    return merged, pred_by_offset


def plot_prediction_summary(
    strain_L1,
    strain_H1,
    pred_by_offset,
    triggers,
    sample_rate,
    output_path,
    title,
    max_points=20000,
):
    """
    Plot the full prepared file:
      - top: prepared L1/H1 strain
      - bottom: model score
      - vertical line: trigger peak
      - shaded region: trigger interval

    The full time range is shown. For readability/file size, the plotted arrays
    are downsampled if they contain more than max_points samples.
    """
    if not pred_by_offset:
        return

    # Use offset 0 for the displayed score if available.
    offset = 0 if 0 in pred_by_offset else sorted(pred_by_offset.keys())[0]
    preds = pred_by_offset[offset]

    if len(preds) == 0:
        return

    # Full time axis for strain.
    n_strain = min(len(strain_L1), len(strain_H1))
    t_strain = np.arange(n_strain) / sample_rate

    # Full time axis for prediction stream.
    t_pred = np.arange(len(preds)) / sample_rate + offset / sample_rate

    # Downsample only for plotting; this still shows the whole file.
    strain_step = max(1, n_strain // max_points)
    pred_step = max(1, len(preds) // max_points)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 6), sharex=True, constrained_layout=True
    )

    ax1.plot(
        t_strain[::strain_step],
        strain_L1[:n_strain:strain_step],
        linewidth=0.5,
        label="L1",
    )
    ax1.plot(
        t_strain[::strain_step],
        strain_H1[:n_strain:strain_step],
        linewidth=0.5,
        alpha=0.8,
        label="H1",
    )
    ax1.set_ylabel("prepared strain")
    ax1.legend(loc="upper right")

    ax2.plot(
        t_pred[::pred_step],
        preds[::pred_step],
        linewidth=0.6,
        label=f"model score, offset={offset}",
    )
    ax2.set_xlabel("time from processed file start [s]")
    ax2.set_ylabel("model score")
    ax2.legend(loc="upper right")

    # Mark triggers on both panels.
    for i, trig in enumerate(triggers):
        interval_label = "trigger interval" if i == 0 else None
        peak_label = "trigger peak" if i == 0 else None

        for ax in (ax1, ax2):
            ax.axvspan(
                trig["left_time_s"],
                trig["right_time_s"],
                color="red",
                alpha=0.15,
                label=interval_label,
            )
            ax.axvline(
                trig["peak_time_s"],
                color="red",
                linestyle="-",
                linewidth=1.2,
                label=peak_label,
            )

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")

    fig.suptitle(title)
    plt.savefig(output_path)
    plt.close(fig)


def compare_noise_files(noise_file_a: str, noise_file_b: str) -> float:
    """
    Placeholder for future noise-similarity metric.

    For now this intentionally returns a constant. Replace with the chosen
    PSD/domain-drift metric later.
    """
    return 0.0


def compare_eval_noise_to_training_noise(
    eval_noise_file: str,
    training_noise_files: Iterable[str],
) -> Tuple[Dict[str, float], float]:
    """
    Placeholder helper for comparing one evaluated noise file to training noise files.
    """
    pairwise = {
        path: compare_noise_files(eval_noise_file, path)
        for path in training_noise_files
    }

    average = float(np.mean(list(pairwise.values()))) if pairwise else float("nan")
    return pairwise, average
