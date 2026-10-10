"""
src/splits.py
-------------
Patient-separated train/val/test splits.

Rationale (for judges):
    Beats from the same patient are highly correlated. If we randomly split
    beats, the model sees the same patient in train and test → data leakage →
    inflated accuracy. Patient-separated splits mimic real deployment.
"""

from pathlib import Path
import random
import numpy as np

from .constants import SPLITS_DIR


def make_patient_split(record_ids: list[str],
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       test_ratio: float = 0.15,
                       seed: int = 42) -> dict:
    """
    Split unique record names into train/val/test.
    Returns {'train': [...], 'val': [...], 'test': [...]}.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "ratios must sum to 1.0"

    records = sorted(set(record_ids))
    rng = random.Random(seed)
    rng.shuffle(records)

    n = len(records)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": records[:n_train],
        "val": records[n_train:n_train + n_val],
        "test": records[n_train + n_val:],
    }


def save_splits(splits: dict, out_dir: str | Path = SPLITS_DIR) -> dict:
    """Write each split to <out_dir>/<name>.txt"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name, records in splits.items():
        p = out_dir / f"{name}.txt"
        p.write_text("\n".join(records), encoding="utf-8")
        paths[name] = str(p)
    return paths


def load_splits(in_dir: str | Path = SPLITS_DIR) -> dict:
    """Read splits from <in_dir>/<name>.txt"""
    in_dir = Path(in_dir)
    out = {}
    for name in ("train", "val", "test"):
        p = in_dir / f"{name}.txt"
        if p.exists():
            out[name] = [r.strip() for r in p.read_text().splitlines() if r.strip()]
    return out


def apply_split(beats: np.ndarray, labels: np.ndarray, record_ids: np.ndarray,
                split: dict) -> dict:
    """
    Return {'train': (X, y), 'val': (X, y), 'test': (X, y)}
    filtered by the record_ids in the split.
    """
    out = {}
    for name, records in split.items():
        mask = np.isin(record_ids, records)
        out[name] = (beats[mask], labels[mask])
    return out


if __name__ == "__main__":
    # quick smoke test with fake record ids
    fake = [f"{i:03d}" for i in range(1, 49)]
    s = make_patient_split(fake)
    print({k: len(v) for k, v in s.items()})
    print("first 3 of each:", {k: v[:3] for k, v in s.items()})