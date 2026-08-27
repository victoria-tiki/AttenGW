#!/usr/bin/env python
"""Run one trained AttenGW model on downloader-produced HDF5 files.

The script loads exactly one checkpoint, scores each input file once, applies
all enabled trigger sweeps, optionally computes one full-file ASD mismatch per
input file, and writes one complete text report per operating point.
"""

import argparse
import glob
import importlib
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from scipy import signal as scipy_signal
from scipy.signal import welch

try:
    from astropy.time import Time
except Exception:
    Time = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_generator import GWDataset, whiten  # noqa: E402


VALIDATION_LOSS_PATTERN = re.compile(
    r"val_loss=([-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
EPOCH_PATTERN = re.compile(r"epoch=(\d+)")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run one AttenGW checkpoint and write one report per trigger operating point."
    )
    parser.add_argument("--config", required=True, help="Inference YAML file.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional output override. The submit script uses this for its job-specific folder.",
    )
    return parser.parse_args()


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The inference config root must be a mapping.")
    return config


def require_mapping(parent, key):
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config entry '{key}' must be a mapping.")
    return value


def finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def decode_hdf5_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.dtype.kind == "S":
        return [decode_hdf5_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def format_number(value):
    return f"{float(value):.12g}"


def filename_number(value):
    return format_number(value).replace("-", "m").replace("+", "p").replace(".", "p")


def gps_to_utc(gps_time):
    if gps_time is None:
        return "unavailable"
    if Time is None:
        return "unavailable (astropy is not installed)"
    try:
        return Time(float(gps_time), format="gps", precision=9).utc.isot + "Z"
    except Exception as error:
        return f"unavailable ({type(error).__name__})"


def validation_loss_from_checkpoint(path):
    match = VALIDATION_LOSS_PATTERN.search(Path(path).name)
    return None if match is None else float(match.group(1))


def epoch_from_checkpoint(path):
    match = EPOCH_PATTERN.search(Path(path).name)
    return None if match is None else int(match.group(1))


def choose_lowest_loss_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    checkpoints = sorted(checkpoint_dir.glob("*.ckpt"))
    candidates = []
    for checkpoint in checkpoints:
        loss = validation_loss_from_checkpoint(checkpoint)
        if loss is None or not np.isfinite(loss):
            continue
        epoch = epoch_from_checkpoint(checkpoint)
        candidates.append((loss, float("inf") if epoch is None else epoch, checkpoint.name, checkpoint))

    if not candidates:
        raise FileNotFoundError(
            f"No .ckpt file in {checkpoint_dir} has a parseable val_loss in its filename."
        )
    return min(candidates)[-1]


def resolve_checkpoint(paths):
    checkpoint_dir = paths.get("checkpoint_dir")
    checkpoint = paths.get("checkpoint")
    checkpoint_dir = None if checkpoint_dir in {None, ""} else checkpoint_dir
    checkpoint = None if checkpoint in {None, ""} else checkpoint

    if (checkpoint_dir is None) == (checkpoint is None):
        raise ValueError("Set exactly one of paths.checkpoint_dir or paths.checkpoint.")

    if checkpoint_dir is not None:
        run_dir = Path(checkpoint_dir).expanduser().resolve()
        if not run_dir.is_dir():
            raise NotADirectoryError(f"Checkpoint directory does not exist: {run_dir}")
        checkpoint_path = choose_lowest_loss_checkpoint(run_dir)
    else:
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        run_dir = checkpoint_path.parent

    training_config_path = run_dir / "config.yaml"
    if not training_config_path.is_file():
        raise FileNotFoundError(
            f"Expected the checkpoint training configuration at {training_config_path}"
        )

    return run_dir, checkpoint_path, training_config_path


def resolve_hdf_files(path_value, description):
    if path_value in {None, ""}:
        raise ValueError(f"{description} path is required.")

    path = Path(str(path_value)).expanduser()
    if path.is_dir():
        files = sorted(set(path.glob("*.hdf5")) | set(path.glob("*.hdf")))
    elif path.is_file():
        files = [path]
    else:
        files = [Path(item) for item in sorted(glob.glob(str(path)))]

    files = [item.resolve() for item in files if item.is_file()]
    if not files:
        raise FileNotFoundError(f"No {description} HDF5 files found for: {path_value}")
    return files


def first_training_value(training_config, choices, label, default=None, required=False):
    for section_name, key in choices:
        section = training_config.get(section_name, {}) or {}
        if key in section and section[key] is not None:
            return section[key]
    if required:
        locations = ", ".join(f"{section}.{key}" for section, key in choices)
        raise KeyError(f"Training config must define {label} in one of: {locations}")
    return default


def training_settings(training_config, inference_config):
    shared = require_mapping(training_config, "shared")
    training = require_mapping(training_config, "training")

    if "noise_is_whitened" not in shared:
        raise KeyError("Training config must define shared.noise_is_whitened.")
    if bool(shared["noise_is_whitened"]):
        raise ValueError(
            "This inference script requires raw training noise: "
            "shared.noise_is_whitened must be false."
        )

    settings = {
        "sample_rate": float(first_training_value(
            training_config, [("shared", "sample_rate")], "sample rate", required=True
        )),
        "band_low": float(first_training_value(
            training_config, [("shared", "band_low")], "low-frequency cutoff", required=True
        )),
        "band_high": float(first_training_value(
            training_config, [("shared", "band_high")], "high-frequency cutoff", required=True
        )),
        "psd_floor": float(first_training_value(
            training_config, [("shared", "psd_floor")], "PSD floor", default=1e-48
        )),
        "psd_outband": float(first_training_value(
            training_config, [("shared", "psd_outband")], "out-of-band PSD value", default=1e40
        )),
        "segment_length": int(first_training_value(
            training_config,
            [("training", "segment_length"), ("shared", "segment_length")],
            "model segment length",
            required=True,
        )),
        "edge_buffer": int(first_training_value(
            training_config,
            [("training", "edge_buffer"), ("shared", "edge_buffer")],
            "whitening edge buffer",
            required=True,
        )),
        "whitening_context_seconds": float(first_training_value(
            training_config,
            [("training", "whitening_context_seconds")],
            "whitening context length",
            required=True,
        )),
        "normalize_per_window_shared": bool(training.get("normalize_per_window_shared", False)),
        "stride": int(inference_config.get("stride", first_training_value(
            training_config,
            [("training", "segment_length"), ("shared", "segment_length")],
            "model segment length",
            required=True,
        ))),
        "offsets": [int(value) for value in inference_config.get("offsets", [0])],
        "batch_size": int(inference_config.get("batch_size", 32)),
    }

    if settings["segment_length"] <= 0 or settings["edge_buffer"] < 0:
        raise ValueError("Training segment length must be positive and edge buffer nonnegative.")
    if settings["whitening_context_seconds"] <= 0:
        raise ValueError("training.whitening_context_seconds must be positive.")
    if settings["stride"] <= 0 or settings["batch_size"] <= 0:
        raise ValueError("inference.stride and inference.batch_size must be positive.")
    if not settings["offsets"] or any(offset < 0 for offset in settings["offsets"]):
        raise ValueError("inference.offsets must contain at least one nonnegative integer.")

    settings["whitening_context_samples"] = int(
        round(settings["whitening_context_seconds"] * settings["sample_rate"])
    )
    minimum_context = settings["segment_length"] + 2 * settings["edge_buffer"]
    if settings["whitening_context_samples"] < minimum_context:
        raise ValueError(
            "The training whitening context is shorter than segment_length + 2*edge_buffer."
        )
    return settings


def clean_model_name(model_name):
    name = str(model_name).strip().replace("\\", "/").rsplit("/", 1)[-1]
    if name.endswith(".py"):
        name = name[:-3]
    if name.startswith("models."):
        name = name[len("models."):]
    if not name.startswith("model_"):
        name = "model_" + name
    return name


def build_model(training_config):
    model_config = require_mapping(training_config, "model")
    model_name = model_config.get("name")
    if not model_name:
        raise KeyError("Training config must define model.name.")
    model_kwargs = model_config.get("kwargs", {}) or {}
    if not isinstance(model_kwargs, dict):
        raise TypeError("Training config model.kwargs must be a mapping.")

    module_name = clean_model_name(model_name)
    module = importlib.import_module(f"models.{module_name}")
    if not hasattr(module, "full_module"):
        raise AttributeError(f"models.{module_name} does not define full_module.")
    return module.full_module(**model_kwargs), module_name, model_kwargs


def checkpoint_state_dict(checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint is not a state-dictionary container.")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict) or not state_dict:
        raise KeyError("Checkpoint does not contain a usable state_dict.")
    return state_dict


def state_dict_variants(state_dict):
    prefixes = ("model.", "module.", "_forward_module.")
    variants = [("unchanged", dict(state_dict))]
    seen = {tuple(state_dict.keys())}
    index = 0
    while index < len(variants):
        description, candidate = variants[index]
        index += 1
        for prefix in prefixes:
            if candidate and all(str(key).startswith(prefix) for key in candidate):
                stripped = {str(key)[len(prefix):]: value for key, value in candidate.items()}
                signature = tuple(stripped.keys())
                if signature not in seen:
                    seen.add(signature)
                    variants.append((f"{description} -> strip {prefix}", stripped))
    return variants


def choose_device(value):
    value = str(value or "auto").lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable.")
    return device


def load_model(checkpoint_path, training_config, device):
    model, model_name, model_kwargs = build_model(training_config)
    raw_state_dict = checkpoint_state_dict(checkpoint_path, device)
    expected = set(model.state_dict().keys())
    variants = sorted(
        state_dict_variants(raw_state_dict),
        key=lambda item: len(set(item[1].keys()).symmetric_difference(expected)),
    )

    errors = []
    for description, candidate in variants:
        try:
            model.load_state_dict(candidate, strict=True)
            model.to(device)
            model.eval()
            return model, model_name, model_kwargs, description
        except RuntimeError as error:
            errors.append((description, str(error)))

    description, error = errors[0]
    raise RuntimeError(
        f"Could not load {checkpoint_path} into {model_name}. "
        f"Closest state-dict variant: {description}.\n{error}"
    )


def parse_trigger_config(config):
    trigger_config = require_mapping(config, "triggers")
    thresholds = sorted({float(value) for value in trigger_config.get("thresholds", [])})
    if not thresholds or any(not np.isfinite(value) or not 0 <= value <= 1 for value in thresholds):
        raise ValueError("triggers.thresholds must contain finite values in [0, 1].")

    merge_tolerance_s = float(trigger_config.get("merge_tolerance_s", 1.0))
    if merge_tolerance_s < 0:
        raise ValueError("triggers.merge_tolerance_s cannot be negative.")

    operating_points = []

    smoothed = require_mapping(trigger_config, "smoothed_max")
    if bool(smoothed.get("enabled", False)):
        smooth_samples = int(smoothed.get("smooth_samples", 64))
        if smooth_samples <= 0:
            raise ValueError("triggers.smoothed_max.smooth_samples must be positive.")
        for threshold in thresholds:
            operating_points.append({
                "method": "smoothed_max",
                "threshold": threshold,
                "smooth_samples": smooth_samples,
                "width": None,
            })

    peak_width = require_mapping(trigger_config, "peak_width")
    if bool(peak_width.get("enabled", False)):
        widths = sorted({int(value) for value in peak_width.get("widths", [])})
        if not widths or any(value <= 0 for value in widths):
            raise ValueError("Enabled peak_width requires positive triggers.peak_width.widths.")
        for threshold in thresholds:
            for width in widths:
                operating_points.append({
                    "method": "peak_width",
                    "threshold": threshold,
                    "width": width,
                })

    full_peak = require_mapping(trigger_config, "full_peak")
    if bool(full_peak.get("enabled", False)):
        widths = sorted({int(value) for value in full_peak.get("widths", [])})
        if not widths or any(value <= 0 for value in widths):
            raise ValueError("Enabled full_peak requires positive triggers.full_peak.widths.")
        mean_margin = float(full_peak.get("mean_margin", 0.05))
        mean_cap = float(full_peak.get("mean_cap", 0.95))
        if mean_margin < 0 or not 0 <= mean_cap <= 1:
            raise ValueError("full_peak mean_margin must be nonnegative and mean_cap in [0, 1].")
        for threshold in thresholds:
            for width in widths:
                operating_points.append({
                    "method": "full_peak",
                    "threshold": threshold,
                    "width": width,
                    "mean_margin": mean_margin,
                    "mean_cap": mean_cap,
                })

    if not operating_points:
        raise ValueError("At least one trigger method must be enabled.")
    return operating_points, merge_tolerance_s


def operating_point_key(point):
    return (point["method"], point["threshold"], point.get("width"))


def operating_point_filename(point):
    name = f"{point['method']}_threshold_{filename_number(point['threshold'])}"
    if point.get("width") is not None:
        name += f"_width_{point['width']}"
    return name + ".txt"


def read_detector_file(path):
    required = ["strain_L1", "strain_H1", "psd_L1", "psd_H1", "freqs"]
    with h5py.File(path, "r") as handle:
        missing = [name for name in required if name not in handle]
        if missing:
            raise KeyError(f"Missing required datasets: {missing}")
        raw_l1 = np.asarray(handle["strain_L1"][:], dtype=np.float64)
        raw_h1 = np.asarray(handle["strain_H1"][:], dtype=np.float64)
        freqs = np.asarray(handle["freqs"][:], dtype=np.float64).squeeze()
        psd_l1 = np.asarray(handle["psd_L1"][:], dtype=np.float64).squeeze()
        psd_h1 = np.asarray(handle["psd_H1"][:], dtype=np.float64).squeeze()
        attributes = {
            key: decode_hdf5_value(value) for key, value in handle.attrs.items()
        }

    if "whiten" not in attributes:
        raise KeyError("Input file must define the HDF5 attribute 'whiten'.")
    if bool(attributes["whiten"]):
        raise ValueError("Input files must contain raw, unwhitened strain (whiten=false).")

    common_length = min(len(raw_l1), len(raw_h1))
    if common_length == 0:
        raise ValueError("Input strain datasets are empty.")
    return (
        raw_l1[:common_length],
        raw_h1[:common_length],
        freqs,
        psd_l1,
        psd_h1,
        attributes,
    )


def metadata_messages(attributes, settings, sanity_config):
    messages = []
    strict = bool(sanity_config.get("strict", True))

    def check(label, actual, expected):
        if actual is None:
            message = f"Missing metadata for {label}; expected {expected}."
        elif not np.isclose(float(actual), float(expected), rtol=0, atol=1e-9):
            message = f"Metadata mismatch for {label}: file={actual}, training={expected}."
        else:
            return
        if strict:
            raise ValueError(message)
        messages.append("WARNING: " + message)

    if bool(sanity_config.get("check_sample_rate", True)):
        check("sample_rate", finite_float(attributes.get("sample_rate")), settings["sample_rate"])
    if bool(sanity_config.get("check_bandpass", True)):
        check("band_low", finite_float(attributes.get("band_low")), settings["band_low"])
        check("band_high", finite_float(attributes.get("band_high")), settings["band_high"])
    return messages


def effective_sample_zero_gps(attributes):
    sample_zero = finite_float(attributes.get("sample_zero_gps"))
    if sample_zero is not None:
        return sample_zero, "sample_zero_gps attribute"

    stored_start = finite_float(attributes.get("gps_start"))
    if stored_start is None:
        return None, "unavailable"

    is_legacy_signal = (
        str(attributes.get("mode", "")).strip().lower() == "signal"
        and str(attributes.get("dataset_split", "")).strip().lower() == "test"
        and "requested_window_len_s" in attributes
    )
    if is_legacy_signal:
        return float(np.floor(stored_start)), "legacy downloader signal fallback: floor(gps_start)"
    return stored_start, "gps_start attribute"


def make_bandlimited_psds(freqs, psd_l1, psd_h1, settings):
    helper = object.__new__(GWDataset)
    helper.dt = 1.0 / settings["sample_rate"]
    helper.psd_floor = settings["psd_floor"]
    helper.psd_outband = settings["psd_outband"]
    helper.band_low = settings["band_low"]
    helper.band_high = settings["band_high"]
    return (
        GWDataset._make_band_limited_psd(helper, freqs, psd_l1),
        GWDataset._make_band_limited_psd(helper, freqs, psd_h1),
    )


def all_window_starts(raw_length, settings):
    usable_length = raw_length - 2 * settings["edge_buffer"]
    last_start = usable_length - settings["segment_length"]
    if last_start < 0:
        return np.array([], dtype=int)

    starts = []
    for offset in sorted(set(settings["offsets"])):
        if offset <= last_start:
            starts.extend(range(offset, last_start + 1, settings["stride"]))
    return np.asarray(sorted(set(starts)), dtype=int)


def choose_context_start(raw_window_start, raw_window_end, raw_length, settings):
    context_length = settings["whitening_context_samples"]
    edge_buffer = settings["edge_buffer"]
    earliest = raw_window_end + edge_buffer - context_length
    latest = raw_window_start - edge_buffer
    if earliest > latest or raw_length < context_length:
        return None

    centered = int(round(0.5 * (raw_window_start + raw_window_end - context_length)))
    context_start = max(earliest, min(centered, latest))
    context_start = max(0, min(context_start, raw_length - context_length))
    return context_start if earliest <= context_start <= latest else None


def normalize_window(l1_window, h1_window, shared_normalization):
    l1_window = np.asarray(l1_window, dtype=np.float64).copy()
    h1_window = np.asarray(h1_window, dtype=np.float64).copy()
    l1_window -= np.mean(l1_window)
    h1_window -= np.mean(h1_window)
    if shared_normalization:
        scale = np.sqrt(0.5 * (np.var(l1_window) + np.var(h1_window))) + 1e-8
        l1_window /= scale
        h1_window /= scale
    return l1_window.astype(np.float32), h1_window.astype(np.float32)


def model_predict_batch(model, l1_batch, h1_batch, device):
    inputs = np.stack([l1_batch, h1_batch], axis=-1).astype(np.float32)
    inputs -= inputs.mean(axis=1, keepdims=True)
    tensor = torch.from_numpy(inputs).to(device)
    with torch.inference_mode():
        predictions = model(tensor)
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    predictions = predictions.detach().cpu().numpy()
    if predictions.ndim == 3:
        predictions = predictions.reshape(predictions.shape[0], -1)
    elif predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    elif predictions.ndim != 2:
        raise ValueError(f"Unexpected model output shape: {predictions.shape}")
    return predictions


def score_file(model, device, raw_l1, raw_h1, psd_l1, psd_h1, settings):
    raw_length = min(len(raw_l1), len(raw_h1))
    starts = all_window_starts(raw_length, settings)
    if len(starts) == 0:
        return np.zeros((0, settings["segment_length"]), dtype=np.float32), starts

    prediction_batches = []
    scored_starts = []
    dt = 1.0 / settings["sample_rate"]

    for batch_index in range(0, len(starts), settings["batch_size"]):
        l1_windows = []
        h1_windows = []
        good_starts = []

        for start in starts[batch_index:batch_index + settings["batch_size"]]:
            raw_start = settings["edge_buffer"] + int(start)
            raw_end = raw_start + settings["segment_length"]
            context_start = choose_context_start(raw_start, raw_end, raw_length, settings)
            if context_start is None:
                continue
            context_end = context_start + settings["whitening_context_samples"]

            white_l1 = whiten.whiten(
                raw_l1[context_start:context_end], psd_l1, dt, floor=settings["psd_floor"]
            )
            white_h1 = whiten.whiten(
                raw_h1[context_start:context_end], psd_h1, dt, floor=settings["psd_floor"]
            )
            local_start = raw_start - context_start
            local_end = local_start + settings["segment_length"]
            l1_window = white_l1[local_start:local_end]
            h1_window = white_h1[local_start:local_end]
            if len(l1_window) != settings["segment_length"] or len(h1_window) != settings["segment_length"]:
                continue

            l1_window, h1_window = normalize_window(
                l1_window, h1_window, settings["normalize_per_window_shared"]
            )
            l1_windows.append(l1_window)
            h1_windows.append(h1_window)
            good_starts.append(int(start))

        if l1_windows:
            prediction_batches.append(
                model_predict_batch(model, np.stack(l1_windows), np.stack(h1_windows), device)
            )
            scored_starts.extend(good_starts)

    if not prediction_batches:
        return np.zeros((0, settings["segment_length"]), dtype=np.float32), np.array([], dtype=int)
    return np.concatenate(prediction_batches), np.asarray(scored_starts, dtype=int)


def moving_average_same(values, width):
    values = np.asarray(values, dtype=float)
    width = int(width)
    if width <= 1 or len(values) == 0:
        return values
    if width % 2 == 0:
        width += 1
    if len(values) < width:
        width = len(values) if len(values) % 2 else max(1, len(values) - 1)
    if width <= 1:
        return values
    return np.convolve(values, np.ones(width) / width, mode="same")


def merge_trigger_samples(samples, tolerance_samples):
    if not samples:
        return []
    samples = sorted(int(value) for value in samples)
    representatives = [samples[0]]
    previous = samples[0]
    for sample in samples[1:]:
        if sample - previous > tolerance_samples:
            representatives.append(sample)
        else:
            representatives[-1] = sample
        previous = sample
    return representatives


def triggers_for_operating_points(predictions, starts, operating_points, settings, merge_tolerance_s):
    """Apply the full threshold/width sweep without repeating model inference."""
    raw = {operating_point_key(point): [] for point in operating_points}
    smoothed_points = [point for point in operating_points if point["method"] == "smoothed_max"]
    width_groups = {}
    for point in operating_points:
        if point.get("width") is not None:
            width_groups.setdefault(point["width"], []).append(point)

    for prediction, start in zip(predictions, starts):
        prediction = np.asarray(prediction, dtype=float).reshape(-1)
        raw_start = settings["edge_buffer"] + int(start)

        if smoothed_points:
            smoothed = moving_average_same(prediction, smoothed_points[0]["smooth_samples"])
            peak = int(np.argmax(smoothed))
            peak_score = float(smoothed[peak])
            for point in smoothed_points:
                if peak_score >= point["threshold"]:
                    raw[operating_point_key(point)].append(raw_start + peak)

        for width, points in width_groups.items():
            peaks, properties = scipy_signal.find_peaks(
                prediction,
                width=int(width),
                distance=int(settings["sample_rate"]),
            )
            peak_info = []
            for peak, left, right in zip(
                peaks, properties.get("left_ips", []), properties.get("right_ips", [])
            ):
                left_index = int(max(0, np.floor(left)))
                right_index = int(min(len(prediction) - 1, np.ceil(right)))
                peak_info.append((int(peak), float(prediction[int(peak)]), prediction[left_index:right_index + 1]))

            for point in points:
                key = operating_point_key(point)
                for peak, peak_score, body in peak_info:
                    if peak_score < point["threshold"]:
                        continue
                    if point["method"] == "full_peak":
                        dynamic_mean = max(
                            0.0,
                            min(point["mean_cap"], point["threshold"] - point["mean_margin"]),
                        )
                        if not body.size or np.mean(body > dynamic_mean) <= 0.5:
                            continue
                    raw[key].append(raw_start + peak)

    tolerance_samples = int(round(merge_tolerance_s * settings["sample_rate"]))
    return {key: merge_trigger_samples(samples, tolerance_samples) for key, samples in raw.items()}

def smooth_feature(values, width):
    width = int(width)
    if width <= 1:
        return values
    if width % 2 == 0:
        width += 1
    return np.convolve(values, np.ones(width) / width, mode="same")


def asd_feature(raw_l1, raw_h1, settings, asd_config):
    nperseg = int(round(float(asd_config.get("welch_seconds", 8.0)) * settings["sample_rate"]))
    if min(len(raw_l1), len(raw_h1)) < nperseg:
        raise ValueError(
            f"File is shorter than the configured Welch segment ({nperseg} samples)."
        )
    overlap = float(asd_config.get("welch_overlap", 0.5))
    if not 0 <= overlap < 1:
        raise ValueError("asd_mismatch.welch_overlap must be in [0, 1).")
    noverlap = int(round(overlap * nperseg))

    freqs, psd_l1 = welch(
        raw_l1,
        fs=settings["sample_rate"],
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )
    freqs_h1, psd_h1 = welch(
        raw_h1,
        fs=settings["sample_rate"],
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )
    if not np.allclose(freqs, freqs_h1):
        raise RuntimeError("Welch frequency grids differ between detectors.")

    mask = (freqs >= settings["band_low"]) & (freqs <= settings["band_high"])
    if not np.any(mask):
        raise ValueError("No Welch bins fall inside the training frequency band.")
    feature = 0.5 * (
        np.log(np.sqrt(np.maximum(psd_l1[mask], 1e-300)))
        + np.log(np.sqrt(np.maximum(psd_h1[mask], 1e-300)))
    )
    return smooth_feature(feature, int(asd_config.get("smooth_bins", 1)))


def build_training_asd_reference(training_files, settings, asd_config, sanity_config):
    features = []
    print(f"Building full-file ASD reference from {len(training_files)} training file(s)...")
    for index, path in enumerate(training_files, start=1):
        print(f"  [{index}/{len(training_files)}] {path.name}", flush=True)
        raw_l1, raw_h1, _, _, _, attributes = read_detector_file(path)
        for message in metadata_messages(attributes, settings, sanity_config):
            print(f"    {message}", flush=True)
        features.append(asd_feature(raw_l1, raw_h1, settings, asd_config))

    shapes = {feature.shape for feature in features}
    if len(shapes) != 1:
        raise ValueError("Training ASD features have inconsistent frequency-grid lengths.")
    k_nearest = int(asd_config.get("k_nearest", 1))
    if k_nearest <= 0 or k_nearest > len(features):
        raise ValueError(
            f"asd_mismatch.k_nearest must be between 1 and {len(features)}."
        )
    return np.stack(features), k_nearest


def compute_asd_mismatch(raw_l1, raw_h1, settings, asd_config, training_features, k_nearest):
    feature = asd_feature(raw_l1, raw_h1, settings, asd_config)
    if feature.shape != training_features.shape[1:]:
        raise ValueError("Evaluation ASD feature does not match the training frequency grid.")
    distances = np.mean(np.abs(training_features - feature[None, :]), axis=1)
    nearest = np.sort(distances)[:k_nearest]
    return float(np.exp(np.mean(nearest)))


def classify_file(record, trigger_samples, event_tolerance_s, sample_rate):
    attributes = record.get("attributes", {})
    mode = str(attributes.get("mode", "")).strip().lower()
    event_gps = finite_float(attributes.get("event_gps"))
    gps_start = record.get("gps_start_used")

    if event_gps is not None and gps_start is not None:
        labels = []
        near_count = 0
        for sample in trigger_samples:
            trigger_gps = gps_start + sample / sample_rate
            near = abs(trigger_gps - event_gps) <= event_tolerance_s
            labels.append("EVENT WINDOW" if near else "FALSE POSITIVE")
            near_count += int(near)
        return {
            "kind": "event",
            "reason": "finite event_gps and usable sample-zero GPS",
            "labels": labels,
            "false_positives": len(trigger_samples) - near_count,
            "unclassified": 0,
            "eligible_event": True,
            "recovered": near_count > 0,
            "event_window_triggers": near_count,
            "metadata_warning": mode == "noise",
        }

    if mode == "noise":
        return {
            "kind": "noise",
            "reason": "mode=noise",
            "labels": ["FALSE POSITIVE"] * len(trigger_samples),
            "false_positives": len(trigger_samples),
            "unclassified": 0,
            "eligible_event": False,
            "recovered": False,
            "event_window_triggers": 0,
            "metadata_warning": False,
        }

    if event_gps is not None:
        reason = "event_gps is present, but no usable GPS timestamp for strain sample zero"
    else:
        reason = "neither finite event_gps nor mode=noise is available"
    return {
        "kind": "unclassified",
        "reason": reason,
        "labels": ["UNCLASSIFIED"] * len(trigger_samples),
        "false_positives": 0,
        "unclassified": len(trigger_samples),
        "eligible_event": False,
        "recovered": False,
        "event_window_triggers": 0,
        "metadata_warning": False,
    }


def display_attribute(attributes, key):
    value = attributes.get(key)
    if value is None or value == "":
        return "unavailable"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def write_report(path, point, records, run_info, event_tolerance_s):
    successful = [record for record in records if record["status"] == "ok"]
    failed = [record for record in records if record["status"] == "failed"]
    key = operating_point_key(point)

    classifications = []
    merged_triggers = 0
    for record in successful:
        triggers = record["triggers"][key]
        classification = classify_file(
            record, triggers, event_tolerance_s, run_info["settings"]["sample_rate"]
        )
        classifications.append((record, triggers, classification))
        merged_triggers += len(triggers)

    false_positives = sum(item[2]["false_positives"] for item in classifications)
    unclassified = sum(item[2]["unclassified"] for item in classifications)
    unclassified_files = sum(item[2]["unclassified"] > 0 for item in classifications)
    eligible_events = sum(item[2]["eligible_event"] for item in classifications)
    recoveries = sum(item[2]["recovered"] for item in classifications)
    classifiable_files = sum(item[2]["kind"] in {"noise", "event"} for item in classifications)

    with open(path, "w", encoding="utf-8") as report:
        report.write("AttenGW inference results\n")
        report.write("=" * 100 + "\n\n")
        report.write(f"Checkpoint directory: {run_info['checkpoint_dir']}\n")
        report.write(f"Checkpoint: {run_info['checkpoint']}\n")
        report.write(
            "Validation loss: "
            + ("unavailable" if run_info["validation_loss"] is None else format_number(run_info["validation_loss"]))
            + "\n"
        )
        report.write(f"Model: {run_info['model_name']}\n")
        report.write(f"Model kwargs: {run_info['model_kwargs']}\n")
        report.write(f"State-dict handling: {run_info['state_dict_handling']}\n")
        report.write(f"Device: {run_info['device']}\n")
        report.write(f"Input path: {run_info['input_path']}\n")
        report.write(f"Files requested: {len(records)}\n")
        report.write(f"Files successfully processed: {len(successful)}\n")
        report.write(f"Files failed: {len(failed)}\n\n")

        report.write(f"Trigger method: {point['method']}\n")
        report.write(f"Threshold: {format_number(point['threshold'])}\n")
        if point.get("width") is not None:
            report.write(f"Width: {point['width']} samples\n")
        if point["method"] == "smoothed_max":
            report.write(f"Smoothing: {point['smooth_samples']} samples\n")
        if point["method"] == "full_peak":
            report.write(f"Mean margin: {format_number(point['mean_margin'])}\n")
            report.write(f"Mean cap: {format_number(point['mean_cap'])}\n")
        report.write(f"Merge tolerance: {format_number(run_info['merge_tolerance_s'])} s\n")
        report.write(f"Event tolerance: {format_number(event_tolerance_s)} s\n\n")

        report.write(f"Merged triggers: {merged_triggers}\n")
        partial_reasons = []
        if unclassified:
            partial_reasons.append(f"{unclassified} trigger(s) unclassified")
        if failed:
            partial_reasons.append(f"{len(failed)} file(s) failed")
        if classifiable_files == 0:
            report.write("False positives: unavailable (no usable noise/event classification metadata)\n")
        elif partial_reasons:
            report.write(f"False positives: {false_positives} (partial; {'; '.join(partial_reasons)})\n")
        else:
            report.write(f"False positives: {false_positives}\n")
        if eligible_events:
            suffix = f" (partial; {len(failed)} file(s) failed)" if failed else ""
            report.write(f"Recoveries: {recoveries}/{eligible_events} eligible event files{suffix}\n")
        else:
            report.write("Recoveries: unavailable (no eligible event files)\n")
        if unclassified:
            report.write(f"Unclassified triggers: {unclassified} from {unclassified_files} file(s)\n")
        report.write("\n")

        report.write("Training preprocessing\n")
        report.write("-" * 100 + "\n")
        for label, setting in [
            ("Sample rate", "sample_rate"),
            ("Band low", "band_low"),
            ("Band high", "band_high"),
            ("Segment length", "segment_length"),
            ("Edge buffer", "edge_buffer"),
            ("Whitening context", "whitening_context_seconds"),
            ("Shared per-window normalization", "normalize_per_window_shared"),
            ("Stride", "stride"),
            ("Offsets", "offsets"),
            ("Batch size", "batch_size"),
        ]:
            report.write(f"{label}: {run_info['settings'][setting]}\n")
        report.write("\n")

        for record in records:
            report.write("#" * 100 + "\n")
            report.write(f"File: {record['path']}\n")
            report.write(f"Status: {record['status'].upper()}\n")
            if record["status"] == "failed":
                report.write(f"Error: {record['error']}\n\n")
                continue

            attributes = record["attributes"]
            triggers = record["triggers"][key]
            classification = classify_file(
                record, triggers, event_tolerance_s, run_info["settings"]["sample_rate"]
            )

            report.write(f"Mode: {display_attribute(attributes, 'mode')}\n")
            report.write(f"Dataset split: {display_attribute(attributes, 'dataset_split')}\n")
            report.write(f"Event names: {display_attribute(attributes, 'event_names')}\n")
            report.write(f"Event GPS: {display_attribute(attributes, 'event_gps')}\n")
            report.write(f"Stored gps_start: {display_attribute(attributes, 'gps_start')}\n")
            report.write(
                "GPS for strain sample zero: "
                + ("unavailable" if record["gps_start_used"] is None else format_number(record["gps_start_used"]))
                + "\n"
            )
            report.write(f"GPS-start handling: {record['gps_start_source']}\n")
            report.write(f"Stored gps_end: {display_attribute(attributes, 'gps_end')}\n")
            report.write(
                "Effective gps_end: "
                + ("unavailable" if record["gps_end_used"] is None else format_number(record["gps_end_used"]))
                + "\n"
            )
            report.write(f"Samples per detector: {record['n_samples']}\n")
            report.write(f"Windows scored: {record['windows_scored']}\n")
            report.write(
                "ASD mismatch: "
                + ("disabled" if record["asd_mismatch"] is None else format_number(record["asd_mismatch"]))
                + "\n"
            )
            for message in record["messages"]:
                report.write(message + "\n")
            if classification["metadata_warning"]:
                report.write(
                    "WARNING: mode=noise conflicts with finite event_gps; event metadata takes precedence.\n"
                )

            report.write(f"Classification: {classification['kind']}\n")
            report.write(f"Classification basis: {classification['reason']}\n")
            if classification["kind"] == "event":
                report.write(f"Recovered: {'yes' if classification['recovered'] else 'no'}\n")
                report.write(f"Event-window triggers: {classification['event_window_triggers']}\n")
                report.write(f"Off-event false positives: {classification['false_positives']}\n")
            elif classification["kind"] == "noise":
                report.write(f"False positives: {classification['false_positives']}\n")
            else:
                report.write(f"Unclassified triggers: {classification['unclassified']}\n")
            report.write(f"Merged triggers in file: {len(triggers)}\n")

            if not triggers:
                report.write("\nNo triggers.\n\n")
                continue

            report.write("\nTriggers\n")
            report.write("-" * 100 + "\n")
            for index, (sample, label) in enumerate(zip(triggers, classification["labels"]), start=1):
                trigger_gps = None
                if record["gps_start_used"] is not None:
                    trigger_gps = record["gps_start_used"] + sample / run_info["settings"]["sample_rate"]
                report.write(f"Trigger {index}:\n")
                report.write(f"  sample index: {sample}\n")
                report.write(
                    "  GPS: " + ("unavailable" if trigger_gps is None else format_number(trigger_gps)) + "\n"
                )
                report.write(f"  UTC: {gps_to_utc(trigger_gps)}\n")
                report.write(f"  classification: {label}\n")
            report.write("\n")


def main():
    args = parse_arguments()
    config = load_yaml(args.config)
    paths = require_mapping(config, "paths")
    inference_config = require_mapping(config, "inference")
    sanity_config = require_mapping(config, "sanity_checks")
    asd_config = require_mapping(config, "asd_mismatch")
    classification_config = require_mapping(config, "classification")

    checkpoint_dir, checkpoint_path, training_config_path = resolve_checkpoint(paths)
    training_config = load_yaml(training_config_path)
    settings = training_settings(training_config, inference_config)
    operating_points, merge_tolerance_s = parse_trigger_config(config)

    event_tolerance_s = float(classification_config.get("event_tolerance_s", 1.0))
    if event_tolerance_s < 0:
        raise ValueError("classification.event_tolerance_s cannot be negative.")

    input_files = resolve_hdf_files(paths.get("input_dir"), "input")
    output_value = args.output_dir or paths.get("output_dir")
    if output_value in {None, ""}:
        raise ValueError("paths.output_dir is required when --output_dir is not supplied.")
    output_dir = Path(output_value).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(inference_config.get("device", "auto"))
    model, model_name, model_kwargs, state_dict_handling = load_model(
        checkpoint_path, training_config, device
    )

    training_features = None
    k_nearest = None
    if bool(asd_config.get("enabled", False)):
        training_files = resolve_hdf_files(paths.get("training_noise_dir"), "training-noise")
        training_features, k_nearest = build_training_asd_reference(
            training_files, settings, asd_config, sanity_config
        )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model: {model_name}")
    print(f"Input files: {len(input_files)}")
    print(f"Operating points: {len(operating_points)}")
    print(f"Output directory: {output_dir}\n")

    records = []
    for index, path in enumerate(input_files, start=1):
        print(f"[{index}/{len(input_files)}] {path.name}", flush=True)
        record = {"path": str(path), "status": "failed", "error": None}
        try:
            raw_l1, raw_h1, freqs, file_psd_l1, file_psd_h1, attributes = read_detector_file(path)
            messages = metadata_messages(attributes, settings, sanity_config)
            gps_start_used, gps_start_source = effective_sample_zero_gps(attributes)
            gps_end_used = (
                None
                if gps_start_used is None
                else gps_start_used + len(raw_l1) / settings["sample_rate"]
            )

            psd_l1, psd_h1 = make_bandlimited_psds(
                freqs, file_psd_l1, file_psd_h1, settings
            )
            predictions, starts = score_file(
                model, device, raw_l1, raw_h1, psd_l1, psd_h1, settings
            )
            if len(predictions) == 0:
                raise RuntimeError("No model windows could be scored.")

            mismatch = None
            if training_features is not None:
                mismatch = compute_asd_mismatch(
                    raw_l1, raw_h1, settings, asd_config, training_features, k_nearest
                )

            record.update({
                "status": "ok",
                "attributes": attributes,
                "messages": messages,
                "n_samples": len(raw_l1),
                "gps_start_used": gps_start_used,
                "gps_start_source": gps_start_source,
                "gps_end_used": gps_end_used,
                "asd_mismatch": mismatch,
                "windows_scored": len(starts),
                "triggers": triggers_for_operating_points(
                    predictions, starts, operating_points, settings, merge_tolerance_s
                ),
            })
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            print(f"  ERROR: {record['error']}", file=sys.stderr, flush=True)
        records.append(record)

    run_info = {
        "checkpoint_dir": checkpoint_dir,
        "checkpoint": checkpoint_path,
        "validation_loss": validation_loss_from_checkpoint(checkpoint_path),
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "state_dict_handling": state_dict_handling,
        "device": device,
        "input_path": paths.get("input_dir"),
        "settings": settings,
        "merge_tolerance_s": merge_tolerance_s,
    }

    print("\nWriting reports:")
    for point in operating_points:
        report_path = output_dir / operating_point_filename(point)
        write_report(report_path, point, records, run_info, event_tolerance_s)
        print(f"  {report_path}")

    if not any(record["status"] == "ok" for record in records):
        raise RuntimeError("All input files failed; reports contain failure details only.")


if __name__ == "__main__":
    main()
