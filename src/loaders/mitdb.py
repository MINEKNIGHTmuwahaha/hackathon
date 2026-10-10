"""
src/loaders/mitdb.py
--------------------
Wraps the existing MIT-BIH loading logic behind the multi-dataset interface.
"""

from pathlib import Path
import numpy as np

from ..data_loader import (
    list_available_records as _list_mitdb,
    load_record as _load_mitdb,
)
from ..preprocess import (
    bandpass_filter, remove_baseline, segment_beats, normalize_beat,
)
from ..dataset import symbol_to_aami
from ..constants import FS_DEFAULT


dataset_id = "mitdb"
FS = FS_DEFAULT  # 360


def list_records() -> list[str]:
    """MIT-BIH record names: '100', '101', ..."""
    return _list_mitdb()


def load_record(name: str):
    """
    Returns:
        beats : (n, 210) float32
        labels : (n,) U1
        r_peaks : (n,) int64
        record_id : str  (e.g. 'mitdb/100')
    """
    signal, r_peaks, ann = _load_mitdb(name)

    filtered = bandpass_filter(signal, fs=FS)
    clean = remove_baseline(filtered, fs=FS)
    beats = segment_beats(clean, r_peaks)

    keep = [i for i, r in enumerate(r_peaks)
            if r - 90 >= 0 and r + 120 <= len(signal)]
    if not keep:
        return None, None, None, None

    beats = beats[:len(keep)]
    ann_kept = [ann[i] for i in keep]
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