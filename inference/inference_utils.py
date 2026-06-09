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
from scipy.stats import wasserstein_distance

# Allow running as: python inference/infer.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib
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


AVAILABLE_MODELS = {
    "model_hdcn_crossattn",
    "model_hdcn_graph",
    "model_tcn_earlyfusion",
    "model_tcn_sepstems_simplefusion",
    "model_tcn_sepstems_interaction",
    "model_tcn_temporalattn_gated",
    "model_tcn_stft",
    "model_bimamba",
}


def build_model_from_training_config(
    training_config: Optional[Dict[str, Any]],
) -> torch.nn.Module:
    if training_config is None:
        raise ValueError(
            "No training config was found in the run directory. "
            "Cannot determine which model architecture to load."
        )

    model_cfg = training_config.get("model", {})
    model_name = model_cfg.get("name", "model_tcn_earlyfusion")
    model_kwargs = model_cfg.get("kwargs", {})

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}' in saved training config. "
            f"Available models are: {', '.join(sorted(AVAILABLE_MODELS))}"
        )

    module = importlib.import_module(f"models.{model_name}")

    if not hasattr(module, "full_module"):
        raise AttributeError(
            f"models.{model_name} does not define full_module"
        )

    print(f"Building model from saved config: {model_name}")
    if model_kwargs:
        print(f"Model kwargs: {model_kwargs}")

    return module.full_module(**model_kwargs)


def load_model_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
    training_config: Optional[Dict[str, Any]],
) -> torch.nn.Module:
    """
    Load the model architecture named in the saved training config, then load
    weights from a Lightning or plain PyTorch checkpoint.

    Lightning checkpoints usually store weights under "state_dict" with a
    "model." prefix, because LightningModel contains self.model = full_module().
    """
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model = build_model_from_training_config(training_config)
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
    shared = (training_config or {}).get("shared", {})
    checkpoint_noise_is_whitened = bool(shared.get("noise_is_whitened", False))

    # For the new raw-noise training path, training mean-centers each 4096-sample
    # window and does not divide by its std. We therefore leave the long stream
    # unnormalized here and apply per-window mean subtraction in WindowDataset.
    #
    # For legacy checkpoints trained with pre-whitened noise, keep std normalization
    # for compatibility with the old data path.
    std_normalization = checkpoint_noise_is_whitened

    if std_normalization:
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
        "checkpoint_noise_is_whitened": checkpoint_noise_is_whitened,
        "whitened_during_inference": not file_whitened,
        "mean_subtraction": "per_window",
        "std_normalization": bool(std_normalization),
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

    def __getitem__(self, idx):
        start = int(self.starts[idx])
        end = start + self.segment_length

        x = np.stack(
            [self.strain_L1[start:end], self.strain_H1[start:end]],
            axis=-1,
        ).astype(np.float32)

        # Match the new/raw training path: mean-center each detector window.
        x[:, 0] -= np.mean(x[:, 0])
        x[:, 1] -= np.mean(x[:, 1])

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
    
def resolve_training_noise_files(training_noise_path: Optional[str]) -> List[str]:
    """
    Resolve optional training-noise path for similarity metrics.

    Accepts:
      - None / "" -> []
      - directory -> sorted *.hdf5 files inside it
      - glob pattern -> sorted matching files
      - single file -> [file]
    """
    if not training_noise_path:
        return []

    path = str(training_noise_path)

    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.hdf5")))

    matches = sorted(glob.glob(path))
    if matches:
        return matches

    if os.path.exists(path):
        return [path]

    print(f"WARNING: no training noise files found for noise_similarity.training_noise_path={path}")
    return []


def _as_float_attr(attrs: Dict[str, Any], key: str, default: float) -> float:
    value = attrs.get(key, default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _smooth_1d(x: np.ndarray, smooth_bins: int) -> np.ndarray:
    smooth_bins = int(smooth_bins)
    if smooth_bins <= 1:
        return x

    # Force odd window length for symmetric smoothing.
    if smooth_bins % 2 == 0:
        smooth_bins += 1

    if len(x) < smooth_bins:
        return x

    kernel = np.ones(smooth_bins, dtype=np.float64) / smooth_bins
    return np.convolve(x, kernel, mode="same")


def _read_noise_psd_features(
    path: str,
    similarity_config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Read saved PSDs and produce smoothed log-ASD curves for H1/L1.

    Uses the HDF5 attrs band_low/band_high when available.
    """
    try:
        data, attrs = read_downloader_hdf5(path)
    except Exception as exc:
        print(f"WARNING: could not read noise file for ASD metrics: {path}: {exc}")
        return None

    required = ["psd_H1", "psd_L1", "freqs"]
    missing = [key for key in required if key not in data]
    if missing:
        print(f"WARNING: skipping ASD features for {path}; missing {missing}")
        return None

    freqs = np.asarray(data["freqs"], dtype=np.float64).squeeze()
    psd_H1 = np.asarray(data["psd_H1"], dtype=np.float64).squeeze()
    psd_L1 = np.asarray(data["psd_L1"], dtype=np.float64).squeeze()

    n = min(len(freqs), len(psd_H1), len(psd_L1))
    freqs = freqs[:n]
    psd_H1 = psd_H1[:n]
    psd_L1 = psd_L1[:n]

    band_low = _as_float_attr(attrs, "band_low", float(np.nanmin(freqs)))
    band_high = _as_float_attr(attrs, "band_high", float(np.nanmax(freqs)))

    psd_floor = 1e-48
    psd_H1 = np.maximum(psd_H1, psd_floor)
    psd_L1 = np.maximum(psd_L1, psd_floor)

    # log-ASD = 0.5 * log(PSD)
    logasd_H1 = 0.5 * np.log(psd_H1)
    logasd_L1 = 0.5 * np.log(psd_L1)

    smooth_bins = int(similarity_config.get("smooth_bins", 9))
    logasd_H1 = _smooth_1d(logasd_H1, smooth_bins)
    logasd_L1 = _smooth_1d(logasd_L1, smooth_bins)

    return {
        "path": path,
        "attrs": attrs,
        "freqs": freqs,
        "band_low": band_low,
        "band_high": band_high,
        "logasd_H1": logasd_H1,
        "logasd_L1": logasd_L1,
    }


def _asd_distance_from_features(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """
    Mean absolute smoothed log-ASD distance, averaged over H1/L1.

    Uses the overlapping HDF5 band attrs and interpolates b onto a's frequency grid.
    """
    low = max(float(a["band_low"]), float(b["band_low"]))
    high = min(float(a["band_high"]), float(b["band_high"]))

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return float("nan")

    freqs_a = np.asarray(a["freqs"])
    freqs_b = np.asarray(b["freqs"])

    mask = (freqs_a >= low) & (freqs_a <= high)
    if np.count_nonzero(mask) < 2:
        return float("nan")

    f = freqs_a[mask]

    distances = []
    for det in ["H1", "L1"]:
        ya = np.asarray(a[f"logasd_{det}"])[mask]
        yb = np.interp(f, freqs_b, np.asarray(b[f"logasd_{det}"]))
        distances.append(float(np.mean(np.abs(ya - yb))))

    return float(np.mean(distances))


def _prepared_tail_window_rate(
    strain_L1: np.ndarray,
    strain_H1: np.ndarray,
    sample_rate: float,
    window_s: float,
    threshold: float,
) -> float:
    """
    Fraction of windows where either detector has max(abs(x)) > threshold.

    This should be computed on the long prepared stream used by inference.
    """
    window_n = int(round(float(window_s) * float(sample_rate)))
    if window_n <= 0:
        return float("nan")

    n = min(len(strain_L1), len(strain_H1))
    n_windows = n // window_n
    if n_windows <= 0:
        return float("nan")

    L = np.asarray(strain_L1[: n_windows * window_n]).reshape(n_windows, window_n)
    H = np.asarray(strain_H1[: n_windows * window_n]).reshape(n_windows, window_n)

    max_abs = np.maximum(np.max(np.abs(L), axis=1), np.max(np.abs(H), axis=1))
    return float(np.mean(max_abs > float(threshold)))


def _prepared_tail_summary(
    strain_L1: np.ndarray,
    strain_H1: np.ndarray,
    sample_rate: float,
    window_s: float,
) -> np.ndarray:
    """
    One transient-amplitude summary per window for optional Wasserstein diagnostic.

    Uses max over detectors of q99.9(|prepared strain|) per window.
    """
    window_n = int(round(float(window_s) * float(sample_rate)))
    if window_n <= 0:
        return np.array([], dtype=np.float64)

    n = min(len(strain_L1), len(strain_H1))
    n_windows = n // window_n
    if n_windows <= 0:
        return np.array([], dtype=np.float64)

    L = np.asarray(strain_L1[: n_windows * window_n]).reshape(n_windows, window_n)
    H = np.asarray(strain_H1[: n_windows * window_n]).reshape(n_windows, window_n)

    qL = np.quantile(np.abs(L), 0.999, axis=1)
    qH = np.quantile(np.abs(H), 0.999, axis=1)
    return np.maximum(qL, qH).astype(np.float64)


def _percentile(value: float, baseline: np.ndarray) -> float:
    baseline = np.asarray(baseline, dtype=np.float64)
    baseline = baseline[np.isfinite(baseline)]

    if not np.isfinite(value) or baseline.size == 0:
        return float("nan")

    return float(100.0 * np.mean(baseline <= value))


def _knn_median(values: Iterable[float], k: int) -> float:
    vals = np.asarray(list(values), dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")

    k_eff = max(1, min(int(k), vals.size))
    return float(np.median(np.sort(vals)[:k_eff]))


def build_training_noise_reference(
    training_noise_files: Iterable[str],
    training_config: Optional[Dict[str, Any]],
    similarity_config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Precompute training-file ASD features and leave-one-out baselines.
    """
    files = list(training_noise_files)
    k = int(similarity_config.get("k_nearest", 3))
    include_wasserstein = bool(similarity_config.get("include_wasserstein", True))
    window_s = float(similarity_config.get("tail_window_s", 1.0))
    threshold = float(similarity_config.get("tail_threshold", 5.0))

    features = []
    for path in files:
        feat = _read_noise_psd_features(path, similarity_config)
        if feat is not None:
            features.append(feat)

    if not features:
        return None

    n = len(features)

    # Pairwise ASD distances.
    asd_matrix = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = _asd_distance_from_features(features[i], features[j])
            asd_matrix[i, j] = d
            asd_matrix[j, i] = d

    # Leave-one-out kNN ASD baseline.
    asd_baseline = []
    for i in range(n):
        row = np.delete(asd_matrix[i], i)
        asd_baseline.append(_knn_median(row, k))
    asd_baseline = np.asarray(asd_baseline, dtype=np.float64)

    # Training tail metrics require preparing strain. This is more expensive but
    # only done once per run and is optional/best-effort.
    tail_rates = []
    tail_summaries = []

    for feat in features:
        path = feat["path"]
        try:
            data, attrs = read_downloader_hdf5(path)
            strain_L1, strain_H1, prep_info = prepare_strain_for_inference(
                data=data,
                attrs=attrs,
                training_config=training_config,
                inference_config={
                    "inference": {
                        "edge_buffer": 2048,
                    }
                },
            )

            sample_rate = float(prep_info["sample_rate"])
            tail_rates.append(
                _prepared_tail_window_rate(
                    strain_L1=strain_L1,
                    strain_H1=strain_H1,
                    sample_rate=sample_rate,
                    window_s=window_s,
                    threshold=threshold,
                )
            )

            if include_wasserstein:
                tail_summaries.append(
                    _prepared_tail_summary(
                        strain_L1=strain_L1,
                        strain_H1=strain_H1,
                        sample_rate=sample_rate,
                        window_s=window_s,
                    )
                )
            else:
                tail_summaries.append(np.array([], dtype=np.float64))

        except Exception as exc:
            print(f"WARNING: could not compute training tail metrics for {path}: {exc}")
            tail_rates.append(float("nan"))
            tail_summaries.append(np.array([], dtype=np.float64))

    tail_rates = np.asarray(tail_rates, dtype=np.float64)

    # Leave-one-out Wasserstein baseline.
    wasserstein_baseline = np.array([], dtype=np.float64)
    if include_wasserstein:
        W = np.full((n, n), np.nan, dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                if tail_summaries[i].size == 0 or tail_summaries[j].size == 0:
                    continue
                d = float(wasserstein_distance(tail_summaries[i], tail_summaries[j]))
                W[i, j] = d
                W[j, i] = d

        wasserstein_baseline = []
        for i in range(n):
            row = np.delete(W[i], i)
            wasserstein_baseline.append(_knn_median(row, k))
        wasserstein_baseline = np.asarray(wasserstein_baseline, dtype=np.float64)

    return {
        "features": features,
        "paths": [f["path"] for f in features],
        "k_nearest": k,
        "asd_baseline": asd_baseline,
        "tail_rates": tail_rates,
        "median_train_tail_window_rate": float(np.nanmedian(tail_rates)),
        "tail_summaries": tail_summaries,
        "wasserstein_baseline": wasserstein_baseline,
        "include_wasserstein": include_wasserstein,
        "tail_window_s": window_s,
        "tail_threshold": threshold,
    }


def compute_noise_domain_metrics(
    eval_noise_file: str,
    eval_strain_L1: np.ndarray,
    eval_strain_H1: np.ndarray,
    eval_attrs: Dict[str, Any],
    training_reference: Optional[Dict[str, Any]],
    training_config: Optional[Dict[str, Any]],
    similarity_config: Dict[str, Any],
    prep_info: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Compute ASD drift and prepared-tail diagnostics for one inference file.

    Returns None if training noise was not configured.
    """
    if training_reference is None:
        return None

    eval_feat = _read_noise_psd_features(eval_noise_file, similarity_config)
    if eval_feat is None:
        return {
            "available": False,
            "message": "Could not compute ASD features for this inference file.",
        }

    k = int(training_reference["k_nearest"])

    # ASD distances to all training files.
    pairwise_asd = {}
    for train_feat in training_reference["features"]:
        d = _asd_distance_from_features(eval_feat, train_feat)
        pairwise_asd[train_feat["path"]] = float(d)

    sorted_pairs = sorted(
        pairwise_asd.items(),
        key=lambda item: item[1] if np.isfinite(item[1]) else np.inf,
    )

    asd_knn_distance = _knn_median(pairwise_asd.values(), k)
    asd_typical_ratio = float(np.exp(asd_knn_distance)) if np.isfinite(asd_knn_distance) else float("nan")
    drift_percentile = _percentile(asd_knn_distance, training_reference["asd_baseline"])

    # Tail-rate diagnostic on prepared inference strain.
    sample_rate = float(prep_info["sample_rate"])
    window_s = float(training_reference["tail_window_s"])
    threshold = float(training_reference["tail_threshold"])

    tail_window_rate = _prepared_tail_window_rate(
        strain_L1=eval_strain_L1,
        strain_H1=eval_strain_H1,
        sample_rate=sample_rate,
        window_s=window_s,
        threshold=threshold,
    )

    median_train_tail = float(training_reference["median_train_tail_window_rate"])
    if np.isfinite(median_train_tail) and median_train_tail > 0:
        tail_rate_ratio = float(tail_window_rate / median_train_tail)
    else:
        tail_rate_ratio = float("nan")

    result = {
        "available": True,
        "training_noise_count": len(training_reference["paths"]),
        "k_nearest": k,
        "asd_knn_distance": float(asd_knn_distance),
        "asd_typical_ratio": float(asd_typical_ratio),
        "drift_percentile": float(drift_percentile),
        "nearest_training_files": [p for p, _ in sorted_pairs[:k]],
        "all_asd_distances_to_training": pairwise_asd,
        "tail_window_rate": float(tail_window_rate),
        "median_train_tail_window_rate": float(median_train_tail),
        "tail_rate_ratio_to_train": float(tail_rate_ratio),
        "tail_threshold": threshold,
        "tail_window_s": window_s,
    }

    if bool(training_reference.get("include_wasserstein", False)):
        eval_tail_summary = _prepared_tail_summary(
            strain_L1=eval_strain_L1,
            strain_H1=eval_strain_H1,
            sample_rate=sample_rate,
            window_s=window_s,
        )

        pairwise_w = {}
        if eval_tail_summary.size > 0:
            for path, train_summary in zip(
                training_reference["paths"],
                training_reference["tail_summaries"],
            ):
                if train_summary.size == 0:
                    pairwise_w[path] = float("nan")
                else:
                    pairwise_w[path] = float(wasserstein_distance(eval_tail_summary, train_summary))

        tail_w_knn = _knn_median(pairwise_w.values(), k)
        tail_w_percentile = _percentile(
            tail_w_knn,
            training_reference.get("wasserstein_baseline", np.array([], dtype=np.float64)),
        )

        result.update(
            {
                "tail_wasserstein_knn": float(tail_w_knn),
                "tail_wasserstein_percentile": float(tail_w_percentile),
                "all_tail_wasserstein_to_training": pairwise_w,
            }
        )

    return result

def format_noise_metrics_text(noise_metrics: Optional[Dict[str, Any]]) -> List[str]:
    lines = []
    lines.append("-" * 100)
    lines.append("NOISE DOMAIN METRICS")
    lines.append("-" * 100)

    if noise_metrics is None:
        lines.append(
            "Training noise files were not provided. "
            "Set noise_similarity.training_noise_path in the inference YAML "
            "to compute ASD drift and tail diagnostics."
        )
        return lines

    if not noise_metrics.get("available", False):
        lines.append(noise_metrics.get("message", "Noise-domain metrics are unavailable."))
        return lines

    asd = float(noise_metrics.get("asd_knn_distance", float("nan")))
    ratio = float(noise_metrics.get("asd_typical_ratio", float("nan")))
    percentile = float(noise_metrics.get("drift_percentile", float("nan")))

    lines.append(f"Training noise files used: {noise_metrics.get('training_noise_count', 'unknown')}")
    lines.append(f"k-nearest aggregation: k={noise_metrics.get('k_nearest', 'unknown')}")
    lines.append("")
    lines.append(f"ASD kNN distance: {asd:.6g}")
    lines.append(f"Approximate ASD mismatch: {ratio:.3g}x")
    lines.append(f"ASD drift percentile: {percentile:.3g}")
    lines.append("")
    lines.append("Interpretation:")

    if np.isfinite(ratio):
        lines.append(
            f"  This file's ASD differs from its nearest training-noise neighborhood "
            f"by about {(ratio - 1.0) * 100.0:.1f}% on average."
        )

    if np.isfinite(percentile):
        lines.append(
            f"  It is more spectrally drifted than about {percentile:.1f}% "
            f"of leave-one-out training-file comparisons."
        )

    tail_rate = float(noise_metrics.get("tail_window_rate", float("nan")))
    median_tail = float(noise_metrics.get("median_train_tail_window_rate", float("nan")))
    tail_ratio = float(noise_metrics.get("tail_rate_ratio_to_train", float("nan")))
    tail_window_s = noise_metrics.get("tail_window_s", "unknown")
    tail_threshold = noise_metrics.get("tail_threshold", "unknown")

    lines.append("")
    lines.append(f"Prepared tail window rate: {tail_rate:.6g}")
    lines.append(f"Median train tail window rate: {median_tail:.6g}")
    lines.append(f"Tail rate ratio to training median: {tail_ratio:.3g}x")
    lines.append(
        f"Tail definition: fraction of {tail_window_s}-s "
        f"windows with max(|prepared strain|) > {tail_threshold}"
    )
    lines.append("")
    lines.append("Interpretation:")

    if np.isfinite(tail_rate):
        lines.append(
            f"  About {100.0 * tail_rate:.2f}% of {tail_window_s}-s windows contain "
            f"a large prepared-strain excursion."
        )

    if np.isfinite(tail_ratio):
        if tail_ratio < 0.5:
            tail_text = "less tail-heavy than the typical training noise file by this diagnostic."
        elif tail_ratio < 1.5:
            tail_text = "broadly similar to the typical training noise file by this diagnostic."
        elif tail_ratio < 3.0:
            tail_text = (
                "more tail-heavy than the typical training noise file; "
                "interpret predictions with some caution."
            )
        else:
            tail_text = (
                "substantially more tail-heavy than the typical training noise file; "
                "interpret predictions with extra caution."
            )

        lines.append(
            f"  This is {tail_ratio:.2g}x the median training-file tail-window rate, "
            f"so this file is {tail_text}"
        )

    if "tail_wasserstein_knn" in noise_metrics:
        tail_w = float(noise_metrics.get("tail_wasserstein_knn", float("nan")))
        tail_w_pct = float(noise_metrics.get("tail_wasserstein_percentile", float("nan")))

        lines.append("")
        lines.append(f"Tail Wasserstein kNN distance: {tail_w:.6g}")
        lines.append(f"Tail Wasserstein percentile: {tail_w_pct:.3g}")
        lines.append("")
        lines.append("Interpretation:")

        if np.isfinite(tail_w_pct):
            if tail_w_pct < 50:
                w_text = (
                    "closer to the training set than a typical leave-one-out "
                    "training comparison."
                )
            elif tail_w_pct < 80:
                w_text = (
                    "within the normal training range by this tail-distribution diagnostic."
                )
            elif tail_w_pct < 95:
                w_text = (
                    "somewhat more tail-distribution shifted than typical training comparisons."
                )
            else:
                w_text = (
                    "unusually tail-distribution shifted relative to the training set; "
                    "interpret predictions with caution."
                )

            lines.append(
                f"  The tail-summary distribution is at the {tail_w_pct:.1f}th percentile "
                f"relative to leave-one-out training comparisons."
            )
            lines.append(f"  It is {w_text}")

    nearest = noise_metrics.get("nearest_training_files", [])
    if nearest:
        lines.append("")
        lines.append("Nearest training files by ASD:")
        for path in nearest:
            lines.append(f"  {path}")

    return lines
    


'''
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
    return pairwise, average'''
