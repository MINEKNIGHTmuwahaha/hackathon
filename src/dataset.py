"""
src/dataset.py
--------------
High-level dataset builder. One function to rule them all:

    build_dataset(...) -> dict with X/y for train/val/test + metadata

This is what Person B/C will import.
"""

from pathlib import Path
import numpy as np

from .constants import (
    AAMI_MAP, AAMI_CLASSES, NON_BEAT_SYMBOLS,
    DATA_DIR,
)
from .data_loader import (
    list_available_records, load_record, load_record_multilead,
)
from .preprocess import (
    bandpass_filter, remove_baseline, segment_beats, normalize_beat,
)
from .splits import make_patient_split, save_splits, load_splits, apply_split
from .logger import get_logger


log = get_logger("sentinel.dataset")


def symbol_to_aami(symbol: str) -> str | None:
    """Map one MIT-BIH symbol to AAMI class, or None if not a beat."""
    if symbol in NON_BEAT_SYMBOLS:
        return None
    for cls, symbols in AAMI_MAP.items():
        if symbol in symbols:
            return cls
    return None


def _extract_beats_from_record(name: str):
    """Return (beats, aami_labels) for one record, aligned."""
    signal, r_peaks, ann = load_record(name)
    filtered = bandpass_filter(signal)
    clean = remove_baseline(filtered)
    beats = segment_beats(clean, r_peaks)

    keep = [i for i, r in enumerate(r_peaks)
            if r - 90 >= 0 and r + 120 <= len(signal)]

    if len(keep) == 0 or len(beats) == 0:
        return None, None

    beats = beats[:len(keep)]
    ann_kept = [ann[i] for i in keep]
    labels = [symbol_to_aami(s) for s in ann_kept]

    # Drop non-beats and align
    final_beats, final_labels = [], []
    for b, l in zip(beats, labels):
        if l is None:
            continue
        final_beats.append(normalize_beat(b))
        final_labels.append(l)

    if not final_beats:
        return None, None

    return np.stack(final_beats), np.array(final_labels, dtype="U1")


def build_dataset(records: list[str] | None = None,
                  cache_path: str | None = None) -> dict:
    """
    Load + preprocess every record, return aligned arrays.
    """
    records = records or list_available_records()
    log.info(f"Building dataset from {len(records)} records")

    all_beats, all_labels, all_ids = [], [], []

    for name in records:
        try:
            beats, labels = _extract_beats_from_record(name)
            if beats is None:
                log.warning(f"{name}: no beats")
                continue
            all_beats.append(beats)
            all_labels.append(labels)
            all_ids.append(np.array([name] * len(beats), dtype="U8"))
            log.info(f"{name}: {len(beats)} beats")
        except Exception as e:
            log.error(f"{name}: {e}")

    if not all_beats:
        raise RuntimeError("No beats loaded from any record")

    X = np.concatenate(all_beats, axis=0)
    y = np.concatenate(all_labels, axis=0)
    ids = np.concatenate(all_ids, axis=0)

    if cache_path:
        p = Path(cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, beats=X.astype(np.float32),
                            labels=y.astype("U1"),
                            record_ids=ids.astype("U8"))
        log.info(f"Cached to {p}")

    return {"X": X, "y": y, "record_ids": ids}


def build_splits(X, y, record_ids, splits_dir="data/splits",
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                 seed=42) -> dict:
    """Split into train/val/test and save split files."""
    split = make_patient_split(list(record_ids),
                               train_ratio, val_ratio, test_ratio, seed)
    save_splits(split, splits_dir)
    log.info(f"Splits saved: { {k: len(v) for k, v in split.items()} }")
    return apply_split(X, y, record_ids, split)


if __name__ == "__main__":
    data = build_dataset(cache_path="data/processed.npz")
    print(f"X: {data['X'].shape}, y: {data['y'].shape}")
    unique, counts = np.unique(data["y"], return_counts=True)
    print("class counts:", dict(zip(unique.tolist(), counts.tolist())))
    splits = build_splits(data["X"], data["y"], data["record_ids"])
    for name, (X_s, y_s) in splits.items():
        print(f"{name}: X={X_s.shape}, y={y_s.shape}")