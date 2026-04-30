import argparse
import numpy as np
from gwosc.datasets import query_events, event_gps
from gwosc.timeline import get_segments
from gwpy.timeseries import TimeSeries
import h5py
import sys
import gc
import os
import matplotlib.pyplot as plt
from astropy.time import Time
import matplotlib.dates as mdates
from scipy.signal import welch
from scipy.signal import welch, butter, filtfilt
from scipy.ndimage import binary_dilation



def parse_args():
    parser = argparse.ArgumentParser(description="Download segments for GW ML dataset")
    parser.add_argument("--gps_start", type=int, required=True,
                        help="GPS start time of overall interval")
    parser.add_argument("--gps_end", type=int, required=True,
                        help="GPS end time of overall interval")
    parser.add_argument("--sample_rate", type=int, default=4096,
                        help="Sampling rate (Hz)")
    parser.add_argument("--window_len_s", type=float, required=True,
                        help="Length of each segment in seconds")
    parser.add_argument("--n_segments", type=int, default=None,
                        help="Number of segments to fetch (None = as many as possible)")
    parser.add_argument("--mode", choices=["noise", "signal", "both"], required=True,
                        help="What kind of segments: noise only, signal only, or both")
    parser.add_argument("--event_pad_s", type=float, default=30.0,
                        help="Padding around each known event to exclude (noise mode)")
    parser.add_argument("--require_full_window", action="store_true",
                        help="If set, only select windows that are full length (= window_len_s).")
    parser.add_argument("--band_low", type=float, default=25.0,
                    help="Low-frequency cutoff used for QC bandpass and plots.")
    parser.add_argument("--band_high", type=float, default=450.0,
                        help="High-frequency cutoff used for QC bandpass and plots.")
    parser.add_argument("--bandpass_order", type=int, default=4,
                        help="Butterworth bandpass filter order.")
    parser.add_argument("--psd_seglen_s", type=float, default=4.0,
                        help="Welch PSD segment length in seconds.")
    parser.add_argument("--target_plot_fs", type=float, default=1024.0,
                        help="Target sample rate for diagnostic time-series plots.")

    # basic scalar cuts on raw / de-glitched strain
    parser.add_argument("--amp_thresh", type=float, default=None,
                        help="Optional amplitude (max|x|) threshold on raw/de-glitched strain.")
    parser.add_argument("--rms_thresh", type=float, default=None,
                        help="Optional RMS threshold on raw/de-glitched strain.")

    # GWOSC data-quality flags
    parser.add_argument(
        "--dq_flags", nargs="+", default=["{ifo}_DATA"],
        help=(
            "GWOSC timeline flags to require for each IFO, "
            "with '{ifo}' placeholder. Example:\n"
            "  --dq_flags '{ifo}_DATA' '{ifo}_CBC_CAT2'\n"
            "Default uses only '{ifo}_DATA'."
        ),
    )

    # aggressive raw-domain cleaning / vetos
    parser.add_argument("--glitch_sigma", type=float, default=None,
                        help="If set, clamp / interpolate samples |x| > glitch_sigma * robust_sigma (MAD-based).")
    parser.add_argument("--glitch_max_frac", type=float, default=0.01,
                        help="If more than this fraction of samples are flagged as glitches, reject window.")
    parser.add_argument("--max_raw_std", type=float, default=None,
                        help="If set, reject window if per-IFO std(raw) after glitch-cleaning exceeds this.")
    parser.add_argument("--min_raw_std", type=float, default=None,
                        help="If set, reject window if per-IFO std(raw) after glitch-cleaning is below this.")
    parser.add_argument("--max_std_ratio", type=float, default=None,
                        help=(
                            "If set, reject window if std(H1)/std(L1) or its inverse "
                            "exceeds this value (i.e., extreme H1/L1 imbalance)."
                        ))

    # plotting / whitening
    parser.add_argument("--plot_timeline", action="store_true",
                        help="If set, plot timeline of selected windows and save as PNG.")
    parser.add_argument("--output_dir", type=str, default="segments_output",
                        help="Directory to save output HDF5 files and plots.")
    parser.add_argument("--plot_timeseries", action="store_true",
                        help="If set, also plot time series for saved noise windows.")
    parser.add_argument("--plot_psd", action="store_true",
                        help="If set, plot example PSDs from saved noise windows.")
    parser.add_argument("--whiten", action="store_true",
                        help="If set, whiten strain before saving (for both noise and signals).")

    return parser.parse_args()


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

def estimate_psd(strain, fs, seglen_s=4.0):
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
        average="mean",
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


def download_and_save_window(
    ifo_list, start, end, sample_rate, mode, output_dir,
    amp_thresh=None, rms_thresh=None, whiten=False,
    glitch_sigma=None, glitch_max_frac=0.01,
    max_raw_std=None, min_raw_std=None, max_std_ratio=None,
    band_low=25.0, band_high=450.0, bandpass_order=4,
    psd_seglen_s=4.0,
):
    """
    Download H1/L1 for [start, end), optionally apply:
      - glitch clamping in full band (raw domain),
      - QC cuts (amp, RMS, std, std-ratio) on a bandpassed copy,
      - whitening using PSD from the cleaned full-band strain.

    Saved HDF5 contains:
      strain_H1, strain_L1  (raw or whitened, depending on `whiten`)
      psd_H1, psd_L1        (PSD estimated from cleaned full-band strain)
      freqs                 (frequency grid for the PSD)

    Attributes: gps_start, gps_end, mode, whiten, sample_rate, band_low, band_high, bandpass_order, psd_seglen_s.

    Returns
    -------
    bool
        True if the window was accepted and saved, False if rejected.
    """
    raw_data   = {}
    proc_data  = {}
    psd_vals   = {}
    freqs_ref  = None
    good       = True
    per_ifo_std = {}    
    reject_reasons = []

    for ifo in ifo_list:
        try:
            ts = TimeSeries.fetch_open_data(
                ifo, int(start), int(end), sample_rate=sample_rate
            )
        except Exception as e:
            reject_reasons.append(f"{ifo}: fetch_open_data failed: {e}")
            good = False
            continue

        vals = ts.value  
        raw_min = np.min(vals)
        raw_max = np.max(vals)
        raw_std = np.std(vals)
        has_nans = np.isnan(vals).any()
        print(
            f"[RAW {mode}] {ifo} {start}-{end}: "
            f"min={raw_min:.3e}, max={raw_max:.3e}, std={raw_std:.3e}, NaNs={has_nans}"
        )

        if has_nans or raw_std < 1e-23:
            reject_reasons.append(
                f"{ifo}: suspicious raw segment "
                f"(std={raw_std:.3e}, NaNs={has_nans})"
            )
            good = False
            continue

        # --- optional glitch clamp on full-band raw data ---
        if glitch_sigma is not None:
            vals_clipped, frac_bad, too_many = clamp_glitches_bp(
                vals,
                fs=sample_rate,
                sigma=glitch_sigma,
                max_frac=glitch_max_frac,
                low=band_low,
                high=band_high,
                order=bandpass_order,
            )
            if too_many:
                reject_reasons.append(
                    f"{ifo}: too_many_glitches (frac_bad={frac_bad:.3e} > {glitch_max_frac:.3e})"
                )
                good = False
            else:
                vals = vals_clipped

        if not good:
            continue

        # --- make a bandpassed copy for QC  ---
        try:
            vals_qc = bandpass_for_qc(
                vals,
                sample_rate,
                low=band_low,
                high=band_high,
                order=bandpass_order,
            )
        except Exception as e:
            reject_reasons.append(f"{ifo}: bandpass_for_qc failed: {e}")
            good = False
            continue

        qc_max = float(np.max(np.abs(vals_qc)))
        qc_rms = float(np.sqrt(np.mean(vals_qc**2)))
        qc_std = float(np.std(vals_qc))

        # store QC std (bandpassed) for cross-IFO ratio checks
        per_ifo_std[ifo] = qc_std

        # --- amplitude / RMS cuts applied on bandpassed copy ---
        if amp_thresh is not None and qc_max > amp_thresh:
            reject_reasons.append(
                f"{ifo}: qc_max={qc_max:.3e} > amp_thresh={amp_thresh:.3e}"
            )
            good = False

        if rms_thresh is not None and qc_rms > rms_thresh:
            reject_reasons.append(
                f"{ifo}: qc_rms={qc_rms:.3e} > rms_thresh={rms_thresh:.3e}"
            )
            good = False

        if not good:
            continue

        # --- PSD from cleaned full-band strain (vals) ---
        freqs_psd, Pxx = estimate_psd(vals, fs=sample_rate, seglen_s=psd_seglen_s)

        # --- whiten or keep raw ---
        if whiten:
            proc = whiten_with_psd(vals, freqs_psd, Pxx, fs=sample_rate)
        else:
            proc = vals

        raw_data[ifo]  = vals
        proc_data[ifo] = proc
        psd_vals[ifo]  = Pxx
        if freqs_ref is None:
            freqs_ref = freqs_psd

    # --- cross-IFO QC std / ratio vetos (on bandpassed qc_std) ---
    if good and (max_raw_std is not None or min_raw_std is not None or max_std_ratio is not None):
        stds = [per_ifo_std.get(ifo, np.nan) for ifo in ifo_list] 

        if max_raw_std is not None:
            if any((s > max_raw_std) for s in stds if np.isfinite(s)):
                reject_reasons.append(
                    f"qc_std > max_raw_std: stds={['%.3e' % s for s in stds]} "
                    f"max_raw_std={max_raw_std:.3e}"
                )
                good = False

        if min_raw_std is not None:
            if any((s < min_raw_std) for s in stds if np.isfinite(s)):
                reject_reasons.append(
                    f"qc_std < min_raw_std: stds={['%.3e' % s for s in stds]} "
                    f"min_raw_std={min_raw_std:.3e}"
                )
                good = False

        if max_std_ratio is not None and len(stds) >= 2:
            s1, s2 = stds[0], stds[1]
            if s1 > 0.0 and s2 > 0.0 and np.isfinite(s1) and np.isfinite(s2):
                ratio = max(s1 / s2, s2 / s1)
                if ratio > max_std_ratio:
                    reject_reasons.append(
                        f"qc_std_ratio={ratio:.2f} > max_std_ratio={max_std_ratio:.2f} "
                        f"(stds={['%.3e' % s for s in stds]})"
                    )
                    good = False

    if not good:
        reason_str = "; ".join(reject_reasons) if reject_reasons else "no_specific_reason"
        print(f"[REJECT] {mode} {start}-{end}: {reason_str}")
        return False

    # here, the window is accepted ---
    whiten_label = "white" if whiten else "raw"
    fname = f"{mode}_{whiten_label}_{int(start)}_{int(end)}.hdf5"
    path  = os.path.join(output_dir, fname)

    with h5py.File(path, "w") as f:
        # processed strain (raw or whitened)
        f.create_dataset("strain_H1", data=proc_data["H1"])
        f.create_dataset("strain_L1", data=proc_data["L1"])

        # PSDs from cleaned full-band strain
        f.create_dataset("psd_H1", data=psd_vals["H1"])
        f.create_dataset("psd_L1", data=psd_vals["L1"])

        if freqs_ref is not None:
            f.create_dataset("freqs", data=freqs_ref)

        f.attrs["gps_start"] = start
        f.attrs["gps_end"]   = end
        f.attrs["mode"]      = mode
        f.attrs["whiten"]    = bool(whiten)
        f.attrs["sample_rate"] = sample_rate
        f.attrs["band_low"] = band_low
        f.attrs["band_high"] = band_high
        f.attrs["bandpass_order"] = bandpass_order
        f.attrs["psd_seglen_s"] = psd_seglen_s

    std_log = ", ".join(
        f"{ifo}:qc_std={per_ifo_std.get(ifo, np.nan):.3e}" for ifo in ifo_list
    )
    print(f"[ACCEPT] {mode} {start}-{end}: {std_log}")
    return True


def plot_timeline_all(gps_start, gps_end, noise_intervals, event_times, output_dir):
    def gps_to_mpl(gps):
        t = Time(gps, format='gps')
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
        t = Time(gps, format='gps')
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


def main():
    args = parse_args()
    ifos = ["H1", "L1"]
    os.makedirs(args.output_dir, exist_ok=True)

    cleaned_intervals_for_plot = []
    noise_windows = []
    signal_windows = []

    ev_list = query_events(select=[f"gps-time >= {args.gps_start}", f"gps-time <= {args.gps_end}"])
    all_event_times_for_plot = [event_gps(ev) for ev in ev_list]

    if args.mode in ["noise", "both"]:
        int_H1 = get_good_data_intervals("H1", args.gps_start, args.gps_end, args.dq_flags)
        int_L1 = get_good_data_intervals("L1", args.gps_start, args.gps_end, args.dq_flags)
        good_int = intersect_intervals(int_H1, int_L1)

        event_windows = get_known_event_windows(args.gps_start, args.gps_end, args.event_pad_s)

        # Carve events out of the coincident intervals
        cleaned = subtract_event_windows(good_int, event_windows)
        cleaned_intervals_for_plot = cleaned

        noise_candidates = pick_windows_from_intervals(
            cleaned, args.window_len_s, require_full=args.require_full_window
        )

        rng = np.random.default_rng()
        rng.shuffle(noise_candidates)

        target = args.n_segments if args.n_segments is not None else len(noise_candidates)

        for (s, e) in noise_candidates:
            if len(noise_windows) >= target:
                break
            downloaded = download_and_save_window(
                ifos, s, e, args.sample_rate, mode="noise",
                output_dir=args.output_dir,
                amp_thresh=args.amp_thresh,
                rms_thresh=args.rms_thresh,
                whiten=args.whiten,
                glitch_sigma=args.glitch_sigma,
                glitch_max_frac=args.glitch_max_frac,
                max_raw_std=args.max_raw_std,
                min_raw_std=args.min_raw_std,
                max_std_ratio=args.max_std_ratio,
                band_low=args.band_low,
                band_high=args.band_high,
                bandpass_order=args.bandpass_order,
                psd_seglen_s=args.psd_seglen_s,
            )
            if downloaded:
                noise_windows.append((s, e))
            else:
                print(f"Skipped window {s}-{e} due to DQ / amplitude / RMS / glitch / std vetos.")

    if args.mode in ["signal", "both"]:
        signal_candidates = []
        for ev in ev_list:
            t0 = event_gps(ev)
            ws = t0 - args.window_len_s / 2
            we = ws + args.window_len_s
            signal_candidates.append((ws, we))

        if args.n_segments is not None:
            signal_candidates = signal_candidates[:args.n_segments]

        for (s, e) in signal_candidates:
            # by default, don't clamp glitches on signals
            downloaded = download_and_save_window(
                ifos, s, e, args.sample_rate, mode="signal",
                output_dir=args.output_dir,
                amp_thresh=None,
                rms_thresh=None,
                whiten=args.whiten,
                glitch_sigma=None,
                glitch_max_frac=args.glitch_max_frac,
                max_raw_std=None,
                min_raw_std=None,
                max_std_ratio=None,
                band_low=args.band_low,
                band_high=args.band_high,
                bandpass_order=args.bandpass_order,
                psd_seglen_s=args.psd_seglen_s,
            )
            if downloaded:
                signal_windows.append((s, e))

    if args.plot_timeline:
        plot_timeline_all(
            args.gps_start, args.gps_end,
            cleaned_intervals_for_plot,
            all_event_times_for_plot,
            args.output_dir
        )
        plot_timeline_saved(
            args.gps_start, args.gps_end,
            noise_windows, signal_windows,
            args.output_dir
        )

    if getattr(args, "plot_timeseries", False):

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
        
        whiten_label = "white" if args.whiten else "raw"

        if noise_windows:
            max_plots_noise = len(noise_windows)  # you can keep "all 15"
            for (s, e) in noise_windows[:max_plots_noise]:
                fname = f"noise_{whiten_label}_{int(s)}_{int(e)}.hdf5"
                path = os.path.join(args.output_dir, fname)
                if not os.path.exists(path):
                    fname_alt = f"noise_{int(s)}_{int(e)}.hdf5"
                    path = os.path.join(args.output_dir, fname_alt)
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
                ylim_H1 = 6.0 * h1_std
                ylim_L1 = 6.0 * l1_std
                
                label=f"{lowcut:g}–{highcut:g} Hz bp"

                fig, (ax1, ax2) = plt.subplots(
                    2, 1, sharex=True, figsize=(8, 5), constrained_layout=True
                )
                ax1.plot(t, h1_bp, label=label, alpha=0.6)
                ax1.set_ylabel("strain (H1)")
                ax1.set_title(f"Noise window starting at GPS {s} ({whiten_label})")
                ax1.set_ylim(-ylim_H1, ylim_H1)
                ax1.legend(loc="upper right")

                ax2.plot(t, l1_bp, label=label, alpha=0.6)
                ax2.set_xlabel("time [s] from window start")
                ax2.set_ylabel("strain (L1)")
                ax2.set_ylim(-ylim_L1, ylim_L1)
                ax2.legend(loc="upper right")

                out_png = os.path.join(
                    args.output_dir, f"noise_ts_bp_{whiten_label}_{int(s)}_{int(e)}.png"
                )
                fig.savefig(out_png)
                plt.close(fig)

                del h1, l1, h1_bp, l1_bp, t, fig, ax1, ax2
                gc.collect()

                print(f"Saved bandpassed noise time series plot to {out_png}")

        if signal_windows:
            max_plots_sig = len(signal_windows)
            for (s, e) in signal_windows[:max_plots_sig]:
                fname = f"signal_{whiten_label}_{int(s)}_{int(e)}.hdf5"
                path = os.path.join(args.output_dir, fname)
                if not os.path.exists(path):
                    fname_alt = f"signal_{int(s)}_{int(e)}.hdf5"
                    path = os.path.join(args.output_dir, fname_alt)
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
                ylim_H1 = 6.0 * h1_std
                ylim_L1 = 6.0 * l1_std
                
                label=f"{lowcut:g}–{highcut:g} Hz bp"

                fig, (ax1, ax2) = plt.subplots(
                    2, 1, sharex=True, figsize=(8, 5), constrained_layout=True
                )
                ax1.plot(t, h1_bp, label=label, alpha=0.6)
                ax1.set_ylabel("strain (H1)")
                ax1.set_title(f"Signal window starting at GPS {s} ({whiten_label})")
                ax1.set_ylim(-ylim_H1, ylim_H1)
                ax1.legend(loc="upper right")

                ax2.plot(t, l1_bp, label=label, alpha=0.6)
                ax2.set_xlabel("time [s] from window start")
                ax2.set_ylabel("strain (L1)")
                ax2.set_ylim(-ylim_L1, ylim_L1)
                ax2.legend(loc="upper right")

                out_png = os.path.join(
                    args.output_dir, f"signal_ts_bp_{whiten_label}_{int(s)}_{int(e)}.png"
                )
                fig.savefig(out_png)
                plt.close(fig)

                del h1, l1, h1_bp, l1_bp, t, fig, ax1, ax2
                gc.collect()

                print(f"Saved bandpassed signal time series plot to {out_png}")

    if getattr(args, "plot_psd", False):
        if noise_windows:
            plot_psd_examples(noise_windows, args.output_dir, args.whiten, mode="noise")
        if signal_windows:
            plot_psd_examples(signal_windows, args.output_dir, args.whiten, mode="signal")



if __name__ == "__main__":
    main()

