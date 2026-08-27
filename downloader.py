import argparse
import numpy as np
from gwosc.datasets import query_events, event_gps
from gwosc.timeline import get_segments
from gwpy.timeseries import TimeSeries
import h5py
import sys
import yaml
import gc
import os
import matplotlib.pyplot as plt
from astropy.time import Time
import matplotlib.dates as mdates
from scipy.signal import welch
from scipy.signal import welch, butter, filtfilt
from scipy.ndimage import binary_dilation



def _add_bool_override(parser, name, dest, help_text):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true", default=None,
                       help=help_text)
    group.add_argument(f"--no_{name}", dest=dest, action="store_false",
                       help=f"Disable: {help_text}")


def parse_args(defaults=None, argv=None):
    parser = argparse.ArgumentParser(description="Download segments for GW ML dataset")

    if defaults is None:
        defaults = {}

    # Shared/common overrides.
    parser.add_argument("--random_seed",type=int,default=67,help="Random seed used when shuffling download candidates.")
    parser.add_argument("--sample_rate", type=int, default=4096,help="Sampling rate (Hz).")
    parser.add_argument("--window_len_s", type=float, default=None,help="Training window length and maximum test file length.")
    parser.add_argument("--min_window_len_s", type=float, default=None,help="Minimum test file length.")
    parser.add_argument("--band_low", type=float, default=25.0,help="Low-frequency cutoff used for QC and plots.")
    parser.add_argument("--band_high", type=float, default=450.0,help="High-frequency cutoff used for QC and plots.")
    parser.add_argument("--bandpass_order", type=int, default=4,help="Butterworth bandpass order.")
    parser.add_argument("--psd_seglen_s", type=float, default=4.0,help="Welch PSD segment length in seconds.")
    parser.add_argument("--target_plot_fs", type=float, default=1024.0, help="Target sample rate for time-series plots.")
    parser.add_argument("--noise_dir", type=str, default=None, help="Root output directory; train/ and test/ are created below it.")

    # Job/range overrides.
    _add_bool_override(parser, "train_noise", "train_noise_enabled",
                       "Run the training-noise download.")
    _add_bool_override(parser, "test_noise", "test_noise_enabled",
                       "Download test noise.")
    _add_bool_override(parser, "test_signal", "test_signal_enabled",
                       "Download test signals.")

    parser.add_argument("--train_gps_start", type=int, default=None)
    parser.add_argument("--train_gps_end", type=int, default=None)
    parser.add_argument("--train_n_segments", type=int, default=None,
                        help="Number of successfully saved training-noise files; omit for all.")
    parser.add_argument("--test_gps_start", type=int, default=None)
    parser.add_argument("--test_gps_end", type=int, default=None)
    parser.add_argument("--test_noise_n_segments", type=int, default=None,
                        help="Number of successfully saved test-noise files; omit for all.")
    parser.add_argument("--test_signal_n_events", type=int, default=None,
                        help="Number of test catalog events to consider; omit for all.")

    # Training-noise cleaning thresholds. In test downloads, these same
    # thresholds are used only to populate would-reject metadata flags.
    parser.add_argument("--event_pad_s", type=float, default=30.0,
                        help="Training noise only: padding around known events.")
    parser.add_argument("--glitch_sigma", type=float, default=None)
    parser.add_argument("--glitch_max_frac", type=float, default=0.01)
    parser.add_argument("--max_std_ratio", type=float, default=None)
    parser.add_argument("--amp_thresh", type=float, default=None)
    parser.add_argument("--rms_thresh", type=float, default=None)
    parser.add_argument("--max_raw_std", type=float, default=None)
    parser.add_argument("--min_raw_std", type=float, default=None)

    # Internal default remains H1_DATA/L1_DATA; it need not be in YAML.
    parser.add_argument("--dq_flags", nargs="+", default=["{ifo}_DATA"])

    _add_bool_override(parser, "plot_timeline", "plot_timeline",
                       "Save timeline plots.")
    _add_bool_override(parser, "plot_timeseries", "plot_timeseries",
                       "Save bandpassed time-series plots.")
    _add_bool_override(parser, "plot_psd", "plot_psd",
                       "Save PSD plots.")

    parser.set_defaults(**defaults)
    return parser.parse_args(argv)


def _optional_int(value, field_name):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer or blank, got {value!r}") from exc


def _required_mapping(parent, key):
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config entry '{key}' must be a mapping.")
    return value


def load_download_config(config_path):
    """Load and validate the restructured downloader configuration."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping.")

    shared = _required_mapping(cfg, "shared")
    paths = _required_mapping(cfg, "paths")
    download = _required_mapping(cfg, "download")
    train_cfg = _required_mapping(download, "train_noise")
    test_cfg = _required_mapping(download, "test")
    test_noise_cfg = _required_mapping(test_cfg, "noise")
    test_signal_cfg = _required_mapping(test_cfg, "signal")

    # Fail explicitly if an old/conflicting structure is still present.
    removed_keys = {
        "gps_start", "gps_end", "n_segments", "mode", "require_full_window",
        "rms_glitch_scope", "allow_signal_window_shrink",
        "min_signal_window_len_s",
        # These now belong under download.train_noise.
        "event_pad_s", "glitch_sigma", "glitch_max_frac",
        "max_std_ratio", "amp_thresh", "rms_thresh",
        "max_raw_std", "min_raw_std",
    }
    present_removed = sorted(k for k in removed_keys if k in download)
    if present_removed:
        raise ValueError(
            "Old download config key(s) are still present: "
            + ", ".join(present_removed)
            + ". Use the train_noise/test structure; training cleaning settings belong "
            "under download.train_noise."
        )

    if "enabled" in test_cfg:
        raise ValueError(
            "download.test.enabled is no longer used. Enable test.noise and/or "
            "test.signal directly."
        )

    train_enabled = bool(train_cfg.get("enabled", False))
    test_noise_enabled = bool(test_noise_cfg.get("enabled", False))
    test_signal_enabled = bool(test_signal_cfg.get("enabled", False))
    test_enabled = test_noise_enabled or test_signal_enabled

    if not train_enabled and not test_enabled:
        raise ValueError(
            "At least one of download.train_noise.enabled, "
            "download.test.noise.enabled, or download.test.signal.enabled "
            "must be true."
        )

    window_len_s = float(download["window_len_s"])
    min_window_len_s = float(download.get("min_window_len_s", 64.0))
    if window_len_s <= 0 or min_window_len_s <= 0:
        raise ValueError("window_len_s and min_window_len_s must be positive.")
    if min_window_len_s > window_len_s:
        raise ValueError("min_window_len_s cannot exceed window_len_s.")

    if train_enabled:
        if train_cfg.get("gps_start") is None or train_cfg.get("gps_end") is None:
            raise ValueError("Enabled train_noise requires gps_start and gps_end.")
        if int(train_cfg["gps_start"]) >= int(train_cfg["gps_end"]):
            raise ValueError("train_noise.gps_start must be earlier than gps_end.")
    if test_enabled:
        if test_cfg.get("gps_start") is None or test_cfg.get("gps_end") is None:
            raise ValueError("Enabled test requires gps_start and gps_end.")
        if int(test_cfg["gps_start"]) >= int(test_cfg["gps_end"]):
            raise ValueError("test.gps_start must be earlier than gps_end.")
    if train_enabled and test_enabled:
        train_start, train_end = int(train_cfg["gps_start"]), int(train_cfg["gps_end"])
        test_start, test_end = int(test_cfg["gps_start"]), int(test_cfg["gps_end"])
        if max(train_start, test_start) < min(train_end, test_end):
            raise ValueError(
                "Enabled train_noise and test GPS ranges overlap. "
                "Use disjoint times to avoid train/test leakage."
            )

    defaults = {
        # Automatic output layout below this existing path setting.
        "noise_dir": paths["noise_dir"],

        # Shared preprocessing.
        "random_seed": int(shared.get("random_seed", 67)),
        "sample_rate": int(shared["sample_rate"]),
        "band_low": float(shared["band_low"]),
        "band_high": float(shared["band_high"]),
        "bandpass_order": int(shared["bandpass_order"]),
        "train_whiten": bool(shared["noise_is_whitened"]),

        # Common download settings.
        "window_len_s": window_len_s,
        "min_window_len_s": min_window_len_s,
        "event_pad_s": float(train_cfg.get("event_pad_s", 30.0)),
        "plot_timeline": bool(download["plot_timeline"]),
        "plot_timeseries": bool(download["plot_timeseries"]),
        "plot_psd": bool(download["plot_psd"]),
        "target_plot_fs": float(download["target_plot_fs"]),
        "psd_seglen_s": float(download["psd_seglen_s"]),
        "glitch_sigma": train_cfg.get("glitch_sigma"),
        "glitch_max_frac": float(train_cfg.get("glitch_max_frac", 0.01)),
        "max_std_ratio": train_cfg.get("max_std_ratio"),
        "amp_thresh": train_cfg.get("amp_thresh"),
        "rms_thresh": train_cfg.get("rms_thresh"),
        "max_raw_std": train_cfg.get("max_raw_std"),
        "min_raw_std": train_cfg.get("min_raw_std"),

        # Enabled jobs and independent ranges/limits.
        "train_noise_enabled": train_enabled,
        "train_gps_start": train_cfg.get("gps_start"),
        "train_gps_end": train_cfg.get("gps_end"),
        "train_n_segments": _optional_int(train_cfg.get("n_segments"), "train_noise.n_segments"),
        "test_enabled": test_enabled,
        "test_gps_start": test_cfg.get("gps_start"),
        "test_gps_end": test_cfg.get("gps_end"),
        "test_noise_enabled": test_noise_enabled,
        "test_noise_n_segments": _optional_int(test_noise_cfg.get("n_segments"), "test.noise.n_segments"),
        "test_signal_enabled": test_signal_enabled,
        "test_signal_n_events": _optional_int(test_signal_cfg.get("n_events"), "test.signal.n_events"),
    }

    return defaults

def get_known_event_windows(gps_start, gps_end, pad_s):
    ev_list = query_events(select=[f"gps-time >= {gps_start}", f"gps-time <= {gps_end}"])
    windows = []
    for ev in ev_list:
        t0 = event_gps(ev)
        windows.append((t0 - pad_s, t0 + pad_s))
    return windows


def interval_overlaps_windows(start, end, windows):
    for w0, w1 in windows:
        if start < w1 and end > w0:
            return True
    return False

def get_unique_events_in_range(gps_start, gps_end, dedup_tol_s=1.0):
    ev_list = query_events(select=[
        f"gps-time >= {gps_start}",
        f"gps-time <= {gps_end}",
    ])

    rows = []
    for ev in ev_list:
        try:
            gps = float(event_gps(ev))
        except Exception as e:
            print(f"WARNING: could not get GPS for {ev}: {e}", flush=True)
            continue
        rows.append({"event": ev, "gps": gps})

    rows = sorted(rows, key=lambda r: r["gps"])

    unique = []
    for row in rows:
        if not unique or abs(row["gps"] - unique[-1]["gps"]) > dedup_tol_s:
            unique.append({
                "gps": row["gps"],
                "events": [row["event"]],
            })
        else:
            unique[-1]["events"].append(row["event"])

    print("GWOSC event records before deduplication:", flush=True)
    for row in rows:
        print(f"  {row['event']:<35} gps={row['gps']:.3f}", flush=True)

    print("Unique physical-event GPS groups after deduplication:", flush=True)
    for i, u in enumerate(unique, start=1):
        names = ", ".join(u["events"])
        print(f"  {i:03d}: gps={u['gps']:.3f} names=[{names}]", flush=True)

    return unique
    
def plot_psd_examples(windows, output_dir, whiten, max_plots=5, mode="noise"):
    """
    For a few saved windows, read psd_H1 / psd_L1 / freqs from the HDF5
    and plot them on log-log axes.

    mode: "noise" or "signal" (determines filename prefix).
    """
    if not windows:
        return

    whiten_label = "white" if whiten else "raw"
    nplot = min(max_plots, len(windows))

    for (s, e) in windows[:nplot]:
        fname = f"{mode}_{whiten_label}_{int(s)}_{int(e)}.hdf5"
        path = os.path.join(output_dir, fname)
        if not os.path.exists(path):
            # fallback to old name if needed
            fname_alt = f"{mode}_{int(s)}_{int(e)}.hdf5"
            path = os.path.join(output_dir, fname_alt)
            if not os.path.exists(path):
                continue

        with h5py.File(path, "r") as f:
            freqs = f["freqs"][:]
            psd_H1 = f["psd_H1"][:]
            psd_L1 = f["psd_L1"][:]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(freqs, psd_H1, label="H1")
        ax.loglog(freqs, psd_L1, label="L1")

        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("PSD [1/Hz]")
        ax.set_title(f"{mode.capitalize()} PSDs for window starting at GPS {int(s)} ({whiten_label})")
        ax.grid(True, which="both", linestyle=":")
        ax.legend()

        out_png = os.path.join(output_dir, f"psd_{mode}_{whiten_label}_{int(s)}_{int(e)}.png")
        fig.savefig(out_png)
        plt.close(fig)
        gc.collect()
        print(f"Saved PSD plot to {out_png}")



def bandpass_for_qc(vals, fs, low=25.0, high=450.0, order=4):
    """
    Make a bandpassed copy of 'vals' for quality-control metrics.

    This does NOT replace the full-band data used for PSD/whitening;
    it is only used for:
      - amp_thresh
      - rms_thresh
      - max_raw_std / min_raw_std
      - max_std_ratio

    so that all vetos are applied in the astrophysical band of interest.
    """
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, vals)


def get_good_data_intervals(detector, gps_start, gps_end, flag_templates):
    """
    Intersect multiple GWOSC timeline flags for a given IFO.

    flag_templates is a list of strings with a '{ifo}' placeholder, e.g.:
      ['{ifo}_DATA', '{ifo}_CBC_CAT2']
    """
    all_intervals = None
    for tmpl in flag_templates:
        flagname = tmpl.format(ifo=detector)
        try:
            segs = get_segments(flagname, gps_start, gps_end)
        except Exception as e:
            print(f"Warning: get_segments failed for {flagname}: {e}")
            segs = []
        if all_intervals is None:
            all_intervals = segs
        else:
            all_intervals = intersect_intervals(all_intervals, segs)

    return all_intervals if all_intervals is not None else []


def intersect_intervals(list1, list2):
    out = []
    i, j = 0, 0
    list1 = sorted(list1)
    list2 = sorted(list2)
    while i < len(list1) and j < len(list2):
        a0, a1 = list1[i]
        b0, b1 = list2[j]
        s = max(a0, b0)
        e = min(a1, b1)
        if s < e:
            out.append((s, e))
        if a1 < b1:
            i += 1
        else:
            j += 1
    return out

def clamp_glitches_bp(vals, fs, sigma=3.5, max_frac=0.002,
                      low=25.0, high=450.0, order=4):
    vals_bp = bandpass_for_qc(vals, fs, low=low, high=high, order=order)

    x = vals_bp - np.median(vals_bp)
    mad = np.median(np.abs(x))
    if mad == 0.0 or not np.isfinite(mad):
        return vals, 0.0, False

    robust_std = 1.4826 * mad
    thresh = sigma * robust_std
    max_x = np.max(np.abs(x))

    mask = np.abs(x) > thresh
    mask = binary_dilation(mask, iterations=2)  # interpolate ±2 samples around each hit
    n_bad = mask.sum()
    frac_bad = n_bad / len(vals)

    print(f"[{fs} Hz] robust_std={robust_std:.3e}, "
          f"thresh={thresh:.3e}, max|x|={max_x:.3e}, "
          f"n_bad={n_bad}, frac_bad={frac_bad:.3e}")
    if n_bad == 0:
        return vals, 0.0, False

    if frac_bad > max_frac:
        # too much of the window is crazy → veto
        return vals, frac_bad, True

    good_idx = np.where(~mask)[0]
    bad_idx  = np.where(mask)[0]
    if len(good_idx) < 2:
        return vals, frac_bad, True

    vals_clipped = vals.copy()
    vals_clipped[mask] = np.interp(bad_idx, good_idx, vals[good_idx])
    return vals_clipped, frac_bad, False


def pick_windows_from_intervals(intervals, window_len, require_full=False):
    """
    Break each interval into non-overlapping windows.
    If require_full=True: only keep exact-length windows.
    If require_full=False: keep last partial window too.
    """
    windows = []
    for s, e in intervals:
        s = float(s)
        e = float(e)
        t = s
        while t < e:
            we = min(t + window_len, e)
            if require_full and (we - t) < window_len:
                break
            windows.append((t, we))
            t = we
    return windows


# ---------- PSD + whitening helpers ----------

def estimate_psd(strain, fs, seglen_s=4.0, average="median"):
    """
    Estimate a one-sided PSD using Welch's method.

    Parameters
    ----------
    strain : 1D np.ndarray
        Raw (or de-glitched) strain time series.
    fs : float
        Sampling frequency [Hz].
    seglen_s : float
        Length of each Welch segment in seconds.

    Returns
    -------
    freqs : 1D np.ndarray
        Frequency grid [Hz].
    Pxx : 1D np.ndarray
        Power spectral density [strain^2 / Hz].
    """
    #  avoid DC leaks
    strain = strain - np.mean(strain)

    nperseg = int(seglen_s * fs)
    nperseg = min(nperseg, len(strain))
    noverlap = nperseg // 2

    freqs, Pxx = welch(
        strain,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        average=average,
    )
    return freqs, Pxx


def whiten_with_psd(strain, psd_freqs, psd_vals, fs):
    """
    Whiten 'strain' using a precomputed PSD (freqs + psd_vals).
    """
    dt = 1.0 / fs
    Nt = len(strain)
    freqs_target = np.fft.rfftfreq(Nt, dt)

    # Interpolate PSD onto the FFT frequencies
    psd_interp = np.interp(freqs_target, psd_freqs, psd_vals)
    psd_interp = np.maximum(psd_interp, 1e-40)  

    hf = np.fft.rfft(strain)
    norm = 1.0 / np.sqrt(1.0 / (dt * 2.0))
    white_hf = hf / np.sqrt(psd_interp) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht


def clamp_glitches(vals, sigma=8.0, max_frac=0.01):
    """
    Robustly clamp short, loud glitches *before* PSD estimation.

    - Estimate robust std via MAD.
    - Mark samples with |x - median(x)| > sigma * robust_std as glitches.
    - If too many samples are bad (fraction > max_frac), mark window as bad.
    - Otherwise, replace glitch samples by linear interpolation in time.

    Returns
    -------
    vals_clipped : np.ndarray
        Possibly modified strain.
    frac_bad : float
        Fraction of samples identified as glitches.
    too_many : bool
        True if frac_bad > max_frac (caller should skip this window).
    """
    x = vals - np.median(vals)
    mad = np.median(np.abs(x))

    if mad == 0.0 or not np.isfinite(mad):
        return vals, 0.0, False

    robust_std = 1.4826 * mad  
    thresh = sigma * robust_std

    mask = np.abs(x) > thresh
    n_bad = mask.sum()
    frac_bad = n_bad / len(vals)

    if n_bad == 0:
        return vals, 0.0, False

    # If a huge fraction of the window, reject it
    if frac_bad > max_frac:
        return vals, frac_bad, True

    # Otherwise interpolate across the spikes
    good_idx = np.where(~mask)[0]
    bad_idx = np.where(mask)[0]

    # edge case
    if len(good_idx) < 2:
        return vals, frac_bad, True

    vals_clipped = vals.copy()
    vals_clipped[mask] = np.interp(bad_idx, good_idx, vals[good_idx])

    return vals_clipped, frac_bad, False


def fetch_raw_window(ifo_list, start, end, sample_rate):
    """Fetch raw strain for all IFOs without applying QC or saving."""
    fetched = {}
    errors = []
    fetch_start = int(start)
    fetch_end = int(end)

    if fetch_end <= fetch_start:
        return None, [f"invalid fetch interval {fetch_start}-{fetch_end}"]

    for ifo in ifo_list:
        try:
            ts = TimeSeries.fetch_open_data(
                ifo, fetch_start, fetch_end, sample_rate=sample_rate
            )
            fetched[ifo] = np.asarray(ts.value)
        except Exception as exc:
            errors.append(f"{ifo}: fetch_open_data failed: {exc}")

    if errors:
        return None, errors

    lengths = {ifo: len(vals) for ifo, vals in fetched.items()}
    if len(set(lengths.values())) != 1:
        return None, [f"H1/L1 length mismatch: {lengths}"]

    return fetched, []


def _safe_float_attr(value):
    return np.nan if value is None else float(value)


def download_and_save_window(
    ifo_list,
    start,
    end,
    sample_rate,
    mode,
    output_dir,
    amp_thresh=None,
    rms_thresh=None,
    whiten=False,
    glitch_sigma=None,
    glitch_max_frac=0.01,
    max_raw_std=None,
    min_raw_std=None,
    max_std_ratio=None,
    band_low=25.0,
    band_high=450.0,
    bandpass_order=4,
    psd_seglen_s=4.0,
    event_gps=None,
    event_names=None,
    requested_window_len_s=None,
    saved_window_len_s=None,
    qc_policy="enforce",
    dataset_split="train",
    preloaded_data=None,
):
    """
    Save one H1/L1 window.

    qc_policy="enforce": preserve the original training-noise behavior:
      clean acceptable glitches and reject windows that fail QC.

    qc_policy="flag_only": save untouched raw strain, calculate the same QC
      measurements, and store whether training QC would have rejected it.
      Only unusable data (fetch failure, non-finite values, or near-zero raw
      variance) prevents saving.
    """
    if qc_policy not in {"enforce", "flag_only"}:
        raise ValueError(f"Unknown qc_policy={qc_policy!r}")
    if qc_policy == "flag_only" and whiten:
        raise ValueError("Test/flag-only windows must be saved raw (whiten=False).")

    if preloaded_data is None:
        fetched, fetch_errors = fetch_raw_window(ifo_list, start, end, sample_rate)
        if fetched is None:
            print(f"[REJECT] {mode} {start}-{end}: {'; '.join(fetch_errors)}")
            return False
    else:
        fetched = {ifo: np.asarray(preloaded_data[ifo]) for ifo in ifo_list}

    raw_data = {}
    proc_data = {}
    psd_vals = {}
    freqs_ref = None
    per_ifo_std = {}
    qc_metrics = {}
    hard_reasons = []
    qc_reasons = []

    for ifo in ifo_list:
        vals_raw = np.asarray(fetched[ifo])
        raw_min = float(np.min(vals_raw)) if len(vals_raw) else np.nan
        raw_max = float(np.max(vals_raw)) if len(vals_raw) else np.nan
        raw_std = float(np.std(vals_raw)) if len(vals_raw) else np.nan
        has_nonfinite = not np.isfinite(vals_raw).all()

        print(
            f"[RAW {mode}] {ifo} {start}-{end}: "
            f"min={raw_min:.3e}, max={raw_max:.3e}, "
            f"std={raw_std:.3e}, NaNs={has_nonfinite}"
        )

        qc_metrics[ifo] = {
            "raw_min": raw_min,
            "raw_max": raw_max,
            "raw_std": raw_std,
            "has_nonfinite": bool(has_nonfinite),
            "glitch_frac": 0.0,
            "glitch_too_many": False,
            "qc_max": np.nan,
            "qc_rms": np.nan,
            "qc_std": np.nan,
        }

        if len(vals_raw) == 0 or has_nonfinite or not np.isfinite(raw_std) or raw_std < 1e-23:
            hard_reasons.append(
                f"{ifo}: suspicious raw segment "
                f"(std={raw_std:.3e}, NaNs={has_nonfinite})"
            )
            continue

        raw_data[ifo] = vals_raw
        vals_for_qc = vals_raw
        vals_for_processing = vals_raw

        if glitch_sigma is not None:
            vals_clipped, frac_bad, too_many = clamp_glitches_bp(
                vals_raw,
                fs=sample_rate,
                sigma=glitch_sigma,
                max_frac=glitch_max_frac,
                low=band_low,
                high=band_high,
                order=bandpass_order,
            )
            qc_metrics[ifo]["glitch_frac"] = float(frac_bad)
            qc_metrics[ifo]["glitch_too_many"] = bool(too_many)

            if too_many:
                qc_reasons.append(
                    f"{ifo}: too_many_glitches "
                    f"(frac_bad={frac_bad:.3e} > {glitch_max_frac:.3e})"
                )
            else:
                # Training uses the cleaned strain. Test remains raw on disk,
                # but downstream diagnostic QC mirrors the training-cleaned copy.
                vals_for_qc = vals_clipped
                if qc_policy == "enforce":
                    vals_for_processing = vals_clipped

        try:
            vals_qc = bandpass_for_qc(
                vals_for_qc,
                sample_rate,
                low=band_low,
                high=band_high,
                order=bandpass_order,
            )
        except Exception as exc:
            hard_reasons.append(f"{ifo}: bandpass_for_qc failed: {exc}")
            continue

        qc_max = float(np.max(np.abs(vals_qc)))
        qc_rms = float(np.sqrt(np.mean(vals_qc ** 2)))
        qc_std = float(np.std(vals_qc))
        per_ifo_std[ifo] = qc_std
        qc_metrics[ifo]["qc_max"] = qc_max
        qc_metrics[ifo]["qc_rms"] = qc_rms
        qc_metrics[ifo]["qc_std"] = qc_std

        if amp_thresh is not None and qc_max > amp_thresh:
            qc_reasons.append(
                f"{ifo}: qc_max={qc_max:.3e} > amp_thresh={amp_thresh:.3e}"
            )
        if rms_thresh is not None and qc_rms > rms_thresh:
            qc_reasons.append(
                f"{ifo}: qc_rms={qc_rms:.3e} > rms_thresh={rms_thresh:.3e}"
            )

        # Training PSD/whitening uses the cleaned strain, exactly as before.
        # Test PSD is descriptive and uses the untouched raw strain.
        psd_source = vals_for_processing if qc_policy == "enforce" else vals_raw
        freqs_psd, pxx = estimate_psd(psd_source, fs=sample_rate, seglen_s=psd_seglen_s,
            average="mean" if qc_policy == "enforce" else "median",
        )

        proc = (
            whiten_with_psd(psd_source, freqs_psd, pxx, fs=sample_rate)
            if whiten else psd_source
        )

        proc_data[ifo] = proc
        psd_vals[ifo] = pxx
        if freqs_ref is None:
            freqs_ref = freqs_psd

    if hard_reasons:
        print(f"[REJECT] {mode} {start}-{end}: {'; '.join(hard_reasons)}")
        return False

    stds = [per_ifo_std.get(ifo, np.nan) for ifo in ifo_list]
    if max_raw_std is not None:
        if any(s > max_raw_std for s in stds if np.isfinite(s)):
            qc_reasons.append(
                f"qc_std > max_raw_std: stds={['%.3e' % s for s in stds]} "
                f"max_raw_std={max_raw_std:.3e}"
            )
    if min_raw_std is not None:
        if any(s < min_raw_std for s in stds if np.isfinite(s)):
            qc_reasons.append(
                f"qc_std < min_raw_std: stds={['%.3e' % s for s in stds]} "
                f"min_raw_std={min_raw_std:.3e}"
            )

    qc_std_ratio = np.nan
    if len(stds) >= 2:
        s1, s2 = stds[0], stds[1]
        if s1 > 0.0 and s2 > 0.0 and np.isfinite(s1) and np.isfinite(s2):
            qc_std_ratio = float(max(s1 / s2, s2 / s1))
            if max_std_ratio is not None and qc_std_ratio > max_std_ratio:
                qc_reasons.append(
                    f"qc_std_ratio={qc_std_ratio:.2f} > "
                    f"max_std_ratio={max_std_ratio:.2f} "
                    f"(stds={['%.3e' % s for s in stds]})"
                )

    qc_would_reject_train = bool(qc_reasons)
    if qc_policy == "enforce" and qc_would_reject_train:
        print(f"[REJECT] {mode} {start}-{end}: {'; '.join(qc_reasons)}")
        return False

    whiten_label = "white" if whiten else "raw"
    fname = f"{mode}_{whiten_label}_{int(start)}_{int(end)}.hdf5"
    path = os.path.join(output_dir, fname)

    with h5py.File(path, "w") as f:
        f.create_dataset("strain_H1", data=proc_data["H1"])
        f.create_dataset("strain_L1", data=proc_data["L1"])
        f.create_dataset("psd_H1", data=psd_vals["H1"])
        f.create_dataset("psd_L1", data=psd_vals["L1"])
        if freqs_ref is not None:
            f.create_dataset("freqs", data=freqs_ref)

        f.attrs["mode"] = mode
        f.attrs["dataset_split"] = dataset_split
        f.attrs["qc_policy"] = qc_policy
        f.attrs["whiten"] = bool(whiten)
        f.attrs["sample_rate"] = sample_rate
        f.attrs["band_low"] = band_low
        f.attrs["band_high"] = band_high
        f.attrs["bandpass_order"] = bandpass_order
        f.attrs["psd_seglen_s"] = psd_seglen_s
        f.attrs["psd_average"] = "median"
        f.attrs["gps_start"] = start
        f.attrs["gps_end"] = end

        if event_gps is not None:
            f.attrs["event_gps"] = float(event_gps)
        if event_names is not None:
            if isinstance(event_names, (list, tuple)):
                f.attrs["event_names"] = ",".join(str(x) for x in event_names)
            else:
                f.attrs["event_names"] = str(event_names)

        if requested_window_len_s is not None:
            f.attrs["requested_window_len_s"] = float(requested_window_len_s)
        if saved_window_len_s is not None:
            f.attrs["saved_window_len_s"] = float(saved_window_len_s)
        if requested_window_len_s is not None and saved_window_len_s is not None:
            shortened = float(saved_window_len_s) < float(requested_window_len_s)
            f.attrs["window_shortened_for_availability"] = bool(shortened)
            # Backward-compatible attribute for existing signal readers.
            if mode == "signal":
                f.attrs["signal_window_shrunk"] = bool(shortened)

        f.attrs["qc_would_reject_train"] = qc_would_reject_train
        f.attrs["qc_reject_reasons"] = "; ".join(qc_reasons)
        f.attrs["qc_std_ratio"] = qc_std_ratio
        f.attrs["qc_glitch_sigma"] = _safe_float_attr(glitch_sigma)
        f.attrs["qc_glitch_max_frac"] = float(glitch_max_frac)
        f.attrs["qc_amp_thresh"] = _safe_float_attr(amp_thresh)
        f.attrs["qc_rms_thresh"] = _safe_float_attr(rms_thresh)
        f.attrs["qc_max_raw_std"] = _safe_float_attr(max_raw_std)
        f.attrs["qc_min_raw_std"] = _safe_float_attr(min_raw_std)
        f.attrs["qc_max_std_ratio"] = _safe_float_attr(max_std_ratio)

        for ifo in ifo_list:
            metrics = qc_metrics[ifo]
            for key, value in metrics.items():
                f.attrs[f"qc_{ifo}_{key}"] = value

    std_log = ", ".join(
        f"{ifo}:qc_std={per_ifo_std.get(ifo, np.nan):.3e}" for ifo in ifo_list
    )
    flag_log = " would_reject_train=True" if qc_would_reject_train else ""
    print(f"[ACCEPT] {mode} {start}-{end}: {std_log}{flag_log}")
    return True


def centered_fallback_lengths(max_window_len_s, min_window_len_s):
    """Return the existing halving sequence, always including the minimum."""
    max_w = float(max_window_len_s)
    min_w = float(min_window_len_s)
    lengths = []
    w = max_w
    while w >= min_w:
        lengths.append(w)
        w *= 0.5
    if not lengths or lengths[-1] != min_w:
        lengths.append(min_w)
    return lengths


def interval_is_contained(start, end, intervals):
    return any(start >= s and end <= e for s, e in intervals)


def download_test_signal_with_availability_fallback(
    ifos,
    event_gps_value,
    event_names,
    good_intervals,
    args,
    output_dir,
):
    """Centered signal fallback based only on availability/finite data."""
    lengths = centered_fallback_lengths(args.window_len_s, args.min_window_len_s)

    for window_len_s in lengths:
        s = float(event_gps_value) - window_len_s / 2.0
        e = s + window_len_s
        print(
            f"Trying test signal window centered on event_gps={event_gps_value:.3f}: "
            f"{s:.1f}–{e:.1f} ({window_len_s:g} s); events={event_names}",
            flush=True,
        )

        if not interval_is_contained(s, e, good_intervals):
            print(
                f"[SHRINK SIGNAL] {window_len_s:g} s is not fully inside "
                "coincident H1/L1 DATA time.",
                flush=True,
            )
            continue

        downloaded = download_and_save_window(
            ifos,
            s,
            e,
            args.sample_rate,
            mode="signal",
            output_dir=output_dir,
            amp_thresh=args.amp_thresh,
            rms_thresh=args.rms_thresh,
            whiten=False,
            glitch_sigma=args.glitch_sigma,
            glitch_max_frac=args.glitch_max_frac,
            max_raw_std=args.max_raw_std,
            min_raw_std=args.min_raw_std,
            max_std_ratio=args.max_std_ratio,
            band_low=args.band_low,
            band_high=args.band_high,
            bandpass_order=args.bandpass_order,
            psd_seglen_s=args.psd_seglen_s,
            event_gps=event_gps_value,
            event_names=event_names,
            requested_window_len_s=args.window_len_s,
            saved_window_len_s=window_len_s,
            qc_policy="flag_only",
            dataset_split="test",
        )

        if downloaded:
            print(
                f"[ACCEPT SIGNAL] event_gps={event_gps_value:.3f}; "
                f"saved centered window length={window_len_s:g} s; "
                f"events={event_names}",
                flush=True,
            )
            return True, s, e, window_len_s

        print(
            f"[SHRINK SIGNAL] {window_len_s:g} s failed availability/finite-data "
            "checks; trying shorter if available.",
            flush=True,
        )

    print(
        f"[REJECT SIGNAL] event_gps={event_gps_value:.3f}; no usable centered "
        f"H1/L1 window found down to {args.min_window_len_s:g} s; "
        f"events={event_names}",
        flush=True,
    )
    return False, None, None, None


def chunk_interval_rebalanced(start, end, max_len_s, min_len_s):
    """
    Split one valid interval into non-overlapping chunks in [min_len_s, max_len_s].

    Full maximum-length chunks are used where possible. If the final remainder
    is shorter than the minimum, enough time is moved from the preceding chunk
    to make the final chunk exactly min_len_s. No usable samples are dropped.
    """
    start = float(start)
    end = float(end)
    max_len_s = float(max_len_s)
    min_len_s = float(min_len_s)
    duration = end - start

    eps = 1e-9
    if duration + eps < min_len_s:
        return []
    if duration <= max_len_s + eps:
        return [(start, end)]

    n_full = int(np.floor(duration / max_len_s))
    remainder = duration - n_full * max_len_s
    chunks = []
    cursor = start

    if remainder < eps:
        for _ in range(n_full):
            chunks.append((cursor, cursor + max_len_s))
            cursor += max_len_s
        return chunks

    if remainder + eps >= min_len_s:
        for _ in range(n_full):
            chunks.append((cursor, cursor + max_len_s))
            cursor += max_len_s
        chunks.append((cursor, end))
        return chunks

    # Short remainder: rebalance only the last full chunk and remainder.
    for _ in range(max(0, n_full - 1)):
        chunks.append((cursor, cursor + max_len_s))
        cursor += max_len_s

    penultimate_len = max_len_s - (min_len_s - remainder)
    if penultimate_len + eps < min_len_s:
        # General fallback for unusual max/min choices: distribute the tail
        # evenly while retaining all samples and respecting the maximum.
        tail_duration = end - cursor
        n_tail = int(np.ceil(tail_duration / max_len_s))
        equal_len = tail_duration / n_tail
        if equal_len + eps < min_len_s:
            return []
        for i in range(n_tail):
            chunk_end = end if i == n_tail - 1 else cursor + equal_len
            chunks.append((cursor, chunk_end))
            cursor = chunk_end
        return chunks

    chunks.append((cursor, cursor + penultimate_len))
    cursor += penultimate_len
    chunks.append((cursor, end))
    return chunks


def joint_finite_intervals(preloaded_data, start, sample_rate, min_len_s):
    """Find maximal contiguous intervals finite in every IFO."""
    arrays = list(preloaded_data.values())
    if not arrays:
        return []
    n = min(len(x) for x in arrays)
    if n == 0:
        return []

    finite = np.ones(n, dtype=bool)
    for vals in arrays:
        finite &= np.isfinite(vals[:n])

    padded = np.concatenate(([False], finite, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)

    intervals = []
    for i0, i1 in zip(starts, ends):
        s = float(start) + i0 / float(sample_rate)
        e = float(start) + i1 / float(sample_rate)
        if e - s >= float(min_len_s):
            intervals.append((s, e, i0, i1))
    return intervals


def _slice_preloaded(preloaded_data, i0, i1):
    return {ifo: vals[i0:i1] for ifo, vals in preloaded_data.items()}


def process_test_noise_chunk(
    ifos,
    start,
    end,
    args,
    output_dir,
    remaining_limit=None,
):
    """
    Save all usable data inside one proposed test-noise chunk.

    Fetch failures are recursively divided to locate available subintervals.
    Successful fetches are split at joint non-finite samples, then rebalanced
    into files bounded by min_window_len_s and window_len_s.
    """
    if remaining_limit is not None and remaining_limit <= 0:
        return []

    duration = float(end) - float(start)
    if duration < args.min_window_len_s:
        return []

    fetched, fetch_errors = fetch_raw_window(ifos, start, end, args.sample_rate)
    if fetched is None:
        print(
            f"[TEST NOISE SPLIT] {start:.1f}-{end:.1f}: "
            f"{'; '.join(fetch_errors)}",
            flush=True,
        )
        if duration < 2.0 * args.min_window_len_s:
            return []

        midpoint = float(start) + duration / 2.0
        saved = process_test_noise_chunk(
            ifos, start, midpoint, args, output_dir, remaining_limit
        )
        if remaining_limit is not None:
            remaining_limit -= len(saved)
        saved.extend(
            process_test_noise_chunk(
                ifos, midpoint, end, args, output_dir, remaining_limit
            )
        )
        return saved

    finite_runs = joint_finite_intervals(
        fetched, start, args.sample_rate, args.min_window_len_s
    )
    if not finite_runs:
        print(
            f"[SKIP TEST NOISE] {start:.1f}-{end:.1f}: no joint finite "
            f"subinterval at least {args.min_window_len_s:g} s",
            flush=True,
        )
        return []

    saved = []
    for run_start, run_end, run_i0, run_i1 in finite_runs:
        run_chunks = chunk_interval_rebalanced(
            run_start,
            run_end,
            args.window_len_s,
            args.min_window_len_s,
        )

        for chunk_start, chunk_end in run_chunks:
            if remaining_limit is not None and len(saved) >= remaining_limit:
                return saved

            rel_i0 = int(round((chunk_start - run_start) * args.sample_rate))
            rel_i1 = int(round((chunk_end - run_start) * args.sample_rate))
            abs_i0 = run_i0 + rel_i0
            abs_i1 = run_i0 + rel_i1
            chunk_data = _slice_preloaded(fetched, abs_i0, abs_i1)

            actual_len_s = len(chunk_data[ifos[0]]) / float(args.sample_rate)
            actual_end = chunk_start + actual_len_s
            if actual_len_s < args.min_window_len_s:
                continue

            downloaded = download_and_save_window(
                ifos,
                chunk_start,
                actual_end,
                args.sample_rate,
                mode="noise",
                output_dir=output_dir,
                amp_thresh=args.amp_thresh,
                rms_thresh=args.rms_thresh,
                whiten=False,
                glitch_sigma=args.glitch_sigma,
                glitch_max_frac=args.glitch_max_frac,
                max_raw_std=args.max_raw_std,
                min_raw_std=args.min_raw_std,
                max_std_ratio=args.max_std_ratio,
                band_low=args.band_low,
                band_high=args.band_high,
                bandpass_order=args.bandpass_order,
                psd_seglen_s=args.psd_seglen_s,
                requested_window_len_s=args.window_len_s,
                saved_window_len_s=actual_len_s,
                qc_policy="flag_only",
                dataset_split="test",
                preloaded_data=chunk_data,
            )
            if downloaded:
                saved.append((chunk_start, actual_end))

    return saved

def plot_timeline_all(gps_start, gps_end, noise_intervals, event_times, output_dir):
    def gps_to_mpl(gps):
        t = Time(gps, format='gps').utc
        return mdates.date2num(t.datetime)

    # Merge intervals so each region is drawn once
    merged_noise = merge_intervals(noise_intervals, tol=0.0)

    start_num = gps_to_mpl(gps_start)
    end_num = gps_to_mpl(gps_end)

    fig, ax = plt.subplots(figsize=(12, 2.2))
    ax.set_xlim(start_num, end_num)
    ax.set_ylim(0, 1)
    ax.set_yticks([])

    # ALL coincident noise time as single, non-overlapping green spans
    for (s, e) in merged_noise:
        ax.axvspan(gps_to_mpl(s), gps_to_mpl(e),
                   facecolor='green', alpha=0.25, edgecolor='none')

    # ALL known events as red lines
    for t0 in event_times:
        ax.axvline(gps_to_mpl(t0), color='red', linestyle='--', alpha=0.8)

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    ax.set_xlabel("UTC date")
    ax.set_title("ALL noise time (green) and ALL known GW events (red)")

    fname = os.path.join(output_dir, f"timeline_ALL_{gps_start}_{gps_end}.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close(fig)
    gc.collect()
    print(f"Saved full-context timeline plot to {fname}")


def subtract_event_windows(intervals, event_windows):
    """
    Given:
      intervals: list of (s,e) coincident science-mode segments
      event_windows: list of (a,b) event±pad windows
    Return:
      list of (s,e) with all [a,b] carved out.
    """
    if not intervals:
        return []

    intervals = sorted(intervals)
    event_windows = sorted(event_windows)

    out = []
    for s, e in intervals:
        pieces = [(s, e)]
        for a, b in event_windows:
            new_pieces = []
            for ps, pe in pieces:
                # no overlap
                if pe <= a or ps >= b:
                    new_pieces.append((ps, pe))
                else:
                    if ps < a:
                        new_pieces.append((ps, a))
                    if pe > b:
                        new_pieces.append((b, pe))
            pieces = new_pieces
            if not pieces:
                break
        out.extend(pieces)
    return out


def merge_intervals(intervals, tol=0.0):
    """
    Merge touching intervals so plots look nicer.
    """
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e + tol:
            merged[-1][1] = max(last_e, e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def plot_timeline_saved(gps_start, gps_end, saved_noise_windows, saved_signal_windows, output_dir):
    """
    Plot ONLY the windows we saved.
    """
    def gps_to_mpl(gps):
        t = Time(gps, format='gps').utc
        return mdates.date2num(t.datetime)

    start_num = gps_to_mpl(gps_start)
    end_num = gps_to_mpl(gps_end)

    fig, ax = plt.subplots(figsize=(12, 2.2))
    ax.set_xlim(start_num, end_num)
    ax.set_ylim(0, 1)
    ax.set_yticks([])

    for (s, e) in saved_noise_windows:
        ax.axvspan(gps_to_mpl(s), gps_to_mpl(e), color='green', alpha=0.35)

    for (s, e) in saved_signal_windows:
        t0 = (s + e) / 2.0
        ax.axvline(gps_to_mpl(t0), color='red', linestyle='--', alpha=0.9)

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    ax.set_xlabel("UTC date")
    ax.set_title("SAVED windows only: noise (green) and signals (red)")

    fname = os.path.join(output_dir, f"timeline_SAVED_{gps_start}_{gps_end}.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close(fig)
    print(f"Saved saved-only timeline plot to {fname}")


def plot_saved_timeseries(windows, output_dir, whiten, mode, args):
    """Original diagnostic time-series plotting, factored for each output dir."""
    if not windows:
        return

    target_fs_plot = args.target_plot_fs
    ds_factor = max(1, int(args.sample_rate // target_fs_plot))
    fs_plot = args.sample_rate / ds_factor
    lowcut, highcut = args.band_low, args.band_high
    nyq_plot = 0.5 * fs_plot
    b_plot, a_plot = butter(
        args.bandpass_order,
        [lowcut / nyq_plot, highcut / nyq_plot],
        btype="band",
    )
    whiten_label = "white" if whiten else "raw"

    for s, e in windows:
        fname = f"{mode}_{whiten_label}_{int(s)}_{int(e)}.hdf5"
        path = os.path.join(output_dir, fname)
        if not os.path.exists(path):
            fname_alt = f"{mode}_{int(s)}_{int(e)}.hdf5"
            path = os.path.join(output_dir, fname_alt)
            if not os.path.exists(path):
                continue

        with h5py.File(path, "r") as f:
            h1 = f["strain_H1"][::ds_factor].astype(np.float32)
            l1 = f["strain_L1"][::ds_factor].astype(np.float32)

        h1_bp = filtfilt(b_plot, a_plot, h1)
        l1_bp = filtfilt(b_plot, a_plot, l1)
        t = np.arange(len(h1_bp), dtype=np.float32) / fs_plot
        h1_std = np.std(h1_bp)
        l1_std = np.std(l1_bp)
        ylim_h1 = 6.0 * h1_std
        ylim_l1 = 6.0 * l1_std
        label = f"{lowcut:g}–{highcut:g} Hz bp"

        fig, (ax1, ax2) = plt.subplots(
            2, 1, sharex=True, figsize=(8, 5), constrained_layout=True
        )
        ax1.plot(t, h1_bp, label=label, alpha=0.6)
        ax1.set_ylabel("strain (H1)")
        ax1.set_title(f"{mode.capitalize()} window starting at GPS {s} ({whiten_label})")
        ax1.set_ylim(-ylim_h1, ylim_h1)
        ax1.legend(loc="upper right")

        ax2.plot(t, l1_bp, label=label, alpha=0.6)
        ax2.set_xlabel("time [s] from window start")
        ax2.set_ylabel("strain (L1)")
        ax2.set_ylim(-ylim_l1, ylim_l1)
        ax2.legend(loc="upper right")

        out_png = os.path.join(
            output_dir, f"{mode}_ts_bp_{whiten_label}_{int(s)}_{int(e)}.png"
        )
        fig.savefig(out_png)
        plt.close(fig)
        del h1, l1, h1_bp, l1_bp, t, fig, ax1, ax2
        gc.collect()
        print(f"Saved bandpassed {mode} time series plot to {out_png}")


def get_coincident_noise_intervals(args, gps_start, gps_end, event_pad_s=None):
    int_h1 = get_good_data_intervals("H1", gps_start, gps_end, args.dq_flags)
    int_l1 = get_good_data_intervals("L1", gps_start, gps_end, args.dq_flags)
    good_int = intersect_intervals(int_h1, int_l1)

    print(f"Found {len(int_h1)} H1 data intervals", flush=True)
    print(f"Found {len(int_l1)} L1 data intervals", flush=True)
    print(f"Found {len(good_int)} coincident H1/L1 intervals", flush=True)

    if event_pad_s is None:
        return good_int, good_int

    event_windows = get_known_event_windows(gps_start, gps_end, event_pad_s)
    print(
        f"Excluding {len(event_windows)} known event windows with "
        f"±{event_pad_s} s padding",
        flush=True,
    )
    cleaned = subtract_event_windows(good_int, event_windows)
    return good_int, cleaned


def run_train_noise(args, ifos, output_dir):
    gps_start = int(args.train_gps_start)
    gps_end = int(args.train_gps_end)
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== TRAIN NOISE DOWNLOAD ===", flush=True)
    print(f"GPS range: {gps_start} to {gps_end}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)

    _, cleaned = get_coincident_noise_intervals(
        args, gps_start, gps_end, event_pad_s=args.event_pad_s
    )
    candidates = pick_windows_from_intervals(
        cleaned, args.window_len_s, require_full=True
    )
    print(f"Found {len(candidates)} full training-noise candidates before shuffling", flush=True)

    rng = np.random.default_rng(args.random_seed)
    rng.shuffle(candidates)
    target = args.train_n_segments if args.train_n_segments is not None else len(candidates)
    saved = []

    for s, e in candidates:
        if len(saved) >= target:
            break
        print(
            f"Trying training-noise window {len(saved) + 1}/{target}: "
            f"{s:.1f}–{e:.1f}",
            flush=True,
        )
        downloaded = download_and_save_window(
            ifos,
            s,
            e,
            args.sample_rate,
            mode="noise",
            output_dir=output_dir,
            amp_thresh=args.amp_thresh,
            rms_thresh=args.rms_thresh,
            whiten=args.train_whiten,
            glitch_sigma=args.glitch_sigma,
            glitch_max_frac=args.glitch_max_frac,
            max_raw_std=args.max_raw_std,
            min_raw_std=args.min_raw_std,
            max_std_ratio=args.max_std_ratio,
            band_low=args.band_low,
            band_high=args.band_high,
            bandpass_order=args.bandpass_order,
            psd_seglen_s=args.psd_seglen_s,
            requested_window_len_s=args.window_len_s,
            saved_window_len_s=args.window_len_s,
            qc_policy="enforce",
            dataset_split="train",
        )
        if downloaded:
            saved.append((s, e))
        else:
            print(
                f"Skipped training window {s}-{e} due to DQ / amplitude / "
                "RMS / glitch / std vetos."
            )

    event_times = [
        event_gps(ev)
        for ev in query_events(select=[f"gps-time >= {gps_start}", f"gps-time <= {gps_end}"])
    ]
    if args.plot_timeline:
        plot_timeline_all(gps_start, gps_end, cleaned, event_times, output_dir)
        plot_timeline_saved(gps_start, gps_end, saved, [], output_dir)
    if args.plot_timeseries:
        plot_saved_timeseries(saved, output_dir, args.train_whiten, "noise", args)
    if args.plot_psd:
        plot_psd_examples(saved, output_dir, args.train_whiten, mode="noise")

    print(f"Saved training-noise windows: {len(saved)}", flush=True)
    return saved


def run_test_download(args, ifos, test_root, noise_output_dir, signal_output_dir):
    gps_start = int(args.test_gps_start)
    gps_end = int(args.test_gps_end)
    os.makedirs(test_root, exist_ok=True)
    if args.test_noise_enabled:
        os.makedirs(noise_output_dir, exist_ok=True)
    if args.test_signal_enabled:
        os.makedirs(signal_output_dir, exist_ok=True)

    print("\n=== TEST DOWNLOAD ===", flush=True)
    print(f"GPS range: {gps_start} to {gps_end}", flush=True)
    print(f"Test root: {test_root}", flush=True)
    
    good_intervals, cleaned_noise_intervals = get_coincident_noise_intervals(
        args, gps_start, gps_end, event_pad_s=args.event_pad_s
    )
    
    event_rows = get_unique_events_in_range(gps_start, gps_end, dedup_tol_s=1.0)
    event_times = [row["gps"] for row in event_rows]

    saved_noise = []
    if args.test_noise_enabled:
        base_chunks = []
        for s, e in cleaned_noise_intervals:
            base_chunks.extend(
                chunk_interval_rebalanced(
                    s, e, args.window_len_s, args.min_window_len_s
                )
            )
        print(
            f"Found {len(base_chunks)} bounded test-noise chunks before "
            "finite-data splitting and shuffling",
            flush=True,
        )
        rng = np.random.default_rng(args.random_seed + 1)
        rng.shuffle(base_chunks)
        target = args.test_noise_n_segments

        for s, e in base_chunks:
            if target is not None and len(saved_noise) >= target:
                break
            remaining = None if target is None else target - len(saved_noise)
            saved_noise.extend(
                process_test_noise_chunk(
                    ifos,
                    s,
                    e,
                    args,
                    noise_output_dir,
                    remaining_limit=remaining,
                )
            )

    saved_signal = []
    if args.test_signal_enabled:
        signal_candidates = event_rows
        if args.test_signal_n_events is not None:
            signal_candidates = signal_candidates[:args.test_signal_n_events]
        print(f"Found {len(signal_candidates)} test signal candidates", flush=True)

        for i, row in enumerate(signal_candidates, start=1):
            event_gps_value = float(row["gps"])
            event_names = row["events"]
            print(
                f"Signal candidate {i}/{len(signal_candidates)}: "
                f"event_gps={event_gps_value:.3f}; events={event_names}",
                flush=True,
            )
            downloaded, saved_s, saved_e, _ = download_test_signal_with_availability_fallback(
                ifos,
                event_gps_value,
                event_names,
                good_intervals,
                args,
                signal_output_dir,
            )
            if downloaded:
                saved_signal.append((saved_s, saved_e))
            else:
                print(
                    f"Skipped signal candidate {i}/{len(signal_candidates)} at "
                    f"event_gps={event_gps_value:.3f}",
                    flush=True,
                )

    if args.plot_timeline:
        plot_timeline_all(
            gps_start,
            gps_end,
            cleaned_noise_intervals,
            event_times,
            test_root,
        )
        plot_timeline_saved(
            gps_start,
            gps_end,
            saved_noise,
            saved_signal,
            test_root,
        )
    if args.plot_timeseries:
        if args.test_noise_enabled:
            plot_saved_timeseries(saved_noise, noise_output_dir, False, "noise", args)
        if args.test_signal_enabled:
            plot_saved_timeseries(saved_signal, signal_output_dir, False, "signal", args)
    if args.plot_psd:
        if args.test_noise_enabled:
            plot_psd_examples(saved_noise, noise_output_dir, False, mode="noise")
        if args.test_signal_enabled:
            plot_psd_examples(saved_signal, signal_output_dir, False, mode="signal")

    print(f"Saved test-noise windows: {len(saved_noise)}", flush=True)
    print(f"Saved test-signal windows: {len(saved_signal)}", flush=True)
    return saved_noise, saved_signal


def validate_runtime_args(args):
    if args.noise_dir is None:
        raise ValueError("paths.noise_dir (or --noise_dir) is required.")
    if args.window_len_s is None or args.min_window_len_s is None:
        raise ValueError("window_len_s and min_window_len_s are required.")
    if args.min_window_len_s <= 0 or args.window_len_s <= 0:
        raise ValueError("window lengths must be positive.")
    if args.min_window_len_s > args.window_len_s:
        raise ValueError("min_window_len_s cannot exceed window_len_s.")
    args.test_enabled = bool(args.test_noise_enabled or args.test_signal_enabled)
    if not args.train_noise_enabled and not args.test_enabled:
        raise ValueError(
            "At least one of train_noise, test.noise, or test.signal must be enabled."
        )
    if args.train_noise_enabled:
        if args.train_gps_start is None or args.train_gps_end is None:
            raise ValueError("Enabled train_noise requires train_gps_start/train_gps_end.")
        if int(args.train_gps_start) >= int(args.train_gps_end):
            raise ValueError("train_gps_start must be earlier than train_gps_end.")
    if args.test_enabled:
        if args.test_gps_start is None or args.test_gps_end is None:
            raise ValueError("Enabled test noise/signal requires test_gps_start/test_gps_end.")
        if int(args.test_gps_start) >= int(args.test_gps_end):
            raise ValueError("test_gps_start must be earlier than test_gps_end.")
    if args.train_noise_enabled and args.test_enabled:
        if int(args.test_gps_start) < int(args.train_gps_end):
            raise ValueError(
                "Test data must begin at or after the end of training data "
                "to avoid train/test leakage."
            )


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None,
                            help="YAML config file using train_noise/test sections.")
    pre_args, remaining_argv = pre_parser.parse_known_args()

    defaults = load_download_config(pre_args.config) if pre_args.config else {}
    args = parse_args(defaults=defaults, argv=remaining_argv)
    args.config = pre_args.config
    validate_runtime_args(args)

    ifos = ["H1", "L1"]
    train_output_dir = os.path.join(args.noise_dir, "train")
    test_root = os.path.join(args.noise_dir, "test")
    test_noise_output_dir = os.path.join(test_root, "noise")
    test_signal_output_dir = os.path.join(test_root, "signal")

    print("Starting downloader", flush=True)
    print(f"Sample rate: {args.sample_rate} Hz", flush=True)
    print(f"Training/max test window length: {args.window_len_s:g} s", flush=True)
    print(f"Minimum test window length: {args.min_window_len_s:g} s", flush=True)
    print(f"Training noise whitened before saving: {args.train_whiten}", flush=True)
    print("Test data whitened before saving: False", flush=True)
    print(
        f"Band/QC range: {args.band_low}–{args.band_high} Hz, "
        f"order {args.bandpass_order}",
        flush=True,
    )
    print(
        f"PSD: Welch {args.psd_seglen_s:g} s segments, 50% overlap, median averaging",
        flush=True,
    )

    train_saved = []
    test_noise_saved = []
    test_signal_saved = []

    if args.train_noise_enabled:
        train_saved = run_train_noise(args, ifos, train_output_dir)

    if args.test_enabled:
        test_noise_saved, test_signal_saved = run_test_download(
            args,
            ifos,
            test_root,
            test_noise_output_dir,
            test_signal_output_dir,
        )

    print("\nDownloader finished", flush=True)
    print(f"Saved training-noise windows: {len(train_saved)}", flush=True)
    print(f"Saved test-noise windows: {len(test_noise_saved)}", flush=True)
    print(f"Saved test-signal windows: {len(test_signal_saved)}", flush=True)


if __name__ == "__main__":
    main()
