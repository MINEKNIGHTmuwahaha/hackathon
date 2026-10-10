"""
src/loaders/svdb.py
-------------------
SVDB (MIT-BIH Supraventricular Arrhythmia Database) loader.

Key differences from MIT-BIH:
  - Sampling rate: 128 Hz (resample to 360 Hz)
  - Leads: ECG1, ECG2 (use ECG1)
  - Record names: 800-899
  - Same AAMI annotation symbols
  - More S beats (~12k vs 2.8k in MIT-BIH)
"""

from pathlib import Path
import numpy as np
import wfdb
from scipy.signal import resample

from ..preprocess import (
    bandpass_filter, remove_baseline, segment_beats, normalize_beat,
)
from ..dataset import symbol_to_aami
from ..constants import FS_DEFAULT
from ..logger import get_logger


log = get_logger("sentinel.svdb")

dataset_id = "svdb"
FS_NATIVE = 128
FS_TARGET = FS_DEFAULT  # 360

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "svdb"


def list_records() -> list[str]:
    if not DATA_DIR.exists():
        return []
    return sorted(p.stem for p in DATA_DIR.glob("*.hea"))


def _resample_signal(signal: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """Resample from fs_in to fs_out. Signal may be 1D or 2D (n, leads)."""
    if fs_in == fs_out:
        return signal
    n_in = signal.shape[0]
    n_out = int(round(n_in * fs_out / fs_in))
    if signal.ndim == 1:
        return resample(signal, n_out)
    return resample(signal, n_out, axis=0)


def load_record(name: str):
    """
    Returns:
        beats      : (n, 210) float32
        labels     : (n,) U1
        r_peaks    : (n,) int64  (in resampled 360 Hz space)
        record_id  : str
    """
    path = str(DATA_DIR / name)
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")

    # Use first lead (ECG1)
    signal = rec.p_signal[:, 0].astype(np.float64)
    signal = np.nan_to_num(signal, nan=0.0)

    # Resample 128 → 360 Hz
    signal = _resample_signal(signal, FS_NATIVE, FS_TARGET)

    # Resample annotation sample positions too
    scale = FS_TARGET / FS_NATIVE
    r_peaks = np.round(ann.sample * scale).astype(np.int64)

    # Filter + segment at 360 Hz
    filtered = bandpass_filter(signal, fs=FS_TARGET)
    clean = remove_baseline(filtered, fs=FS_TARGET)
    beats = segment_beats(clean, r_peaks)

    keep = [i for i, r in enumerate(r_peaks)
            if r - 90 >= 0 and r + 120 <= len(signal)]
    if not keep:
        return None, None, None, None

    beats = beats[:len(keep)]
    ann_kept = [ann.symbol[i] for i in keep]
    rp_kept = r_peaks[keep]
    labels = [symbol_to_aami(s) for s in ann_kept]

    final_beats, final_labels, final_rp = [], [], []
    for b, l, r in zip(beats, labels, rp_kept):
        if l is None:
            continue
        final_beats.append(normalize_beat(b))
        final_labels.append(l)
        final_rp.append(r)

    if not final_beats:
        return None, None, None, None

    return (
        np.stack(final_beats).astype(np.float32),
        np.array(final_labels, dtype="U1"),
        np.array(final_rp, dtype=np.int64),
        f"{dataset_id}/{name}",
    )


if __name__ == "__main__":
    records = list_records()
    print(f"SVDB: {len(records)} records available")
    if records:
        beats, labels, rp, rid = load_record(records[0])
        if beats is not None:
            print(f"{rid}: {beats.shape[0]} beats, shape {beats.shape}")
            from collections import Counter
            print(f"class counts: {dict(Counter(labels.tolist()))}")