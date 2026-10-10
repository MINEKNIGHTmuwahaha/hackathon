"""
src/preprocess.py
-----------------
Filtering and beat segmentation for MIT-BIH ECG signals.

Supports both single-lead (1D) and multi-lead (2D) input.
"""

import numpy as np
from scipy.signal import butter, filtfilt, medfilt


def bandpass_filter(signal: np.ndarray, fs: int = 360,
                    low: float = 0.5, high: float = 40.0) -> np.ndarray:
    """
    Butterworth bandpass filter.
    Works on 1D (single lead) or 2D (n_samples, n_leads) arrays.
    """
    nyq = 0.5 * fs
    b, a = butter(N=4, Wn=[low / nyq, high / nyq], btype="band")

    if signal.ndim == 1:
        return filtfilt(b, a, signal)

    # 2D: filter each lead independently
    out = np.empty_like(signal, dtype=np.float64)
    for lead in range(signal.shape[1]):
        out[:, lead] = filtfilt(b, a, signal[:, lead])
    return out


def remove_baseline(signal: np.ndarray, fs: int = 360) -> np.ndarray:
    """Two-stage median filter — 1D or 2D."""
    kernel1 = int(0.2 * fs) | 1
    kernel2 = int(0.6 * fs) | 1

    if signal.ndim == 1:
        baseline = medfilt(medfilt(signal, kernel1), kernel2)
        return signal - baseline

    out = np.empty_like(signal, dtype=np.float64)
    for lead in range(signal.shape[1]):
        baseline = medfilt(medfilt(signal[:, lead], kernel1), kernel2)
        out[:, lead] = signal[:, lead] - baseline
    return out


def segment_beats(signal: np.ndarray, r_peaks: np.ndarray,
                  before: int = 90, after: int = 120) -> np.ndarray:
    """
    Cut a fixed window around each R-peak.

    - 1D input → returns (n_beats, 210)
    - 2D input → returns (n_beats, 210, n_leads)
    """
    window = before + after
    beats = []

    for r in r_peaks:
        start = r - before
        end = r + after
        if start < 0 or end > len(signal):
            continue
        beats.append(signal[start:end])

    if not beats:
        shape = (0, window) if signal.ndim == 1 else (0, window, signal.shape[1])
        return np.empty(shape, dtype=np.float64)

    return np.stack(beats).astype(np.float64)


def normalize_beat(beat: np.ndarray) -> np.ndarray:
    """
    Z-score normalize a single beat.
    Handles 1D (210,) or 2D (210, n_leads).
    """
    if beat.ndim == 1:
        m, s = beat.mean(), beat.std()
        return beat - m if s < 1e-8 else (beat - m) / s

    # 2D: normalize each lead independently
    out = np.empty_like(beat, dtype=np.float64)
    for lead in range(beat.shape[1]):
        col = beat[:, lead]
        m, s = col.mean(), col.std()
        out[:, lead] = col - m if s < 1e-8 else (col - m) / s
    return out


if __name__ == "__main__":
    from .data_loader import load_record, load_record_multilead
    ...

    # Single lead
    signal, r_peaks, _ = load_record("100")
    filtered = bandpass_filter(signal)
    clean = remove_baseline(filtered)
    beats = segment_beats(clean, r_peaks)
    print(f"[1-lead] Filtered: {filtered.shape}")
    print(f"[1-lead] Beats: {beats.shape}")

    # Multi-lead
    signal2, r_peaks2, _ = load_record_multilead("100")
    filtered2 = bandpass_filter(signal2)
    clean2 = remove_baseline(filtered2)
    beats2 = segment_beats(clean2, r_peaks2)
    print(f"[2-lead] Filtered: {filtered2.shape}")
    print(f"[2-lead] Beats: {beats2.shape}")
    print(f"[2-lead] Normalized first beat mean per lead: "
          f"{normalize_beat(beats2[0]).mean(axis=0)}")