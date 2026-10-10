"""
src/demo.py
-----------
Demo-ready beat sampler + prediction strip visualizer.

Usage (from Person C):
    from src.demo import sample_demo_beats, demo_predict
    demo_beats = sample_demo_beats(npz_path="data/processed.npz", per_class=1)
    demo_predict(model, demo_beats)   # -> prints a pretty table
"""

import numpy as np

from .data_loader import load_record
from .preprocess import (
    bandpass_filter, remove_baseline, segment_beats, normalize_beat,
)
from .constants import AAMI_CLASSES
from .logger import get_logger


log = get_logger("sentinel.demo")


def sample_demo_beats(npz_path: str = "data/processed.npz",
                      per_class: int = 1,
                      seed: int = 7) -> dict:
    """
    Pick a small, dramatic set of beats — one per AAMI class by default.

    Returns
    -------
    {
        "X": (n_picked, 210) float32,
        "y": (n_picked,) U1,
        "record_ids": (n_picked,) U8,
        "indices": (n_picked,) int64,  # original indices into the .npz
    }
    """
    d = np.load(npz_path, allow_pickle=True)
    X, y, ids = d["beats"], d["labels"], d["record_ids"]

    rng = np.random.default_rng(seed)
    picked_X, picked_y, picked_ids, picked_idx = [], [], [], []

    for cls in AAMI_CLASSES:
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            log.warning(f"{cls}: no beats available")
            continue
        chosen = rng.choice(idx, size=min(per_class, len(idx)), replace=False)
        for c in chosen:
            picked_X.append(X[c])
            picked_y.append(y[c])
            picked_ids.append(ids[c])
            picked_idx.append(int(c))

    return {
        "X": np.stack(picked_X),
        "y": np.array(picked_y, dtype="U1"),
        "record_ids": np.array(picked_ids, dtype="U8"),
        "indices": np.array(picked_idx, dtype=np.int64),
    }


def demo_predict(classifier, demo: dict) -> dict:
    """
    Run the classifier on demo beats and print a pretty table.

    `classifier` must expose .predict(X) -> (labels, confidences).
    """
    X = demo["X"]
    y_true = demo["y"]

    y_pred, y_conf = classifier.predict(X)

    print("\n" + "=" * 72)
    print(f"{'#':>3} {'record':>7} {'true':>5} {'pred':>5} "
          f"{'conf':>6} {'refer?':>7}")
    print("-" * 72)

    referrals = 0
    for i, (yt, yp, cf, rid) in enumerate(zip(y_true, y_pred, y_conf,
                                              demo["record_ids"])):
        needs_referral = (cf < 0.70) or (yp in ("V", "S", "F", "Q"))
        referrals += int(needs_referral)
        mark = "⚠️ " if needs_referral else "  "
        print(f"{i+1:>3} {rid:>7} {yt:>5} {yp:>5} {cf:>6.3f} "
              f"{mark}{'YES' if needs_referral else 'no':>5}")

    print("=" * 72)
    print(f"Referral rate: {referrals}/{len(y_true)}")

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_conf": y_conf,
        "referrals": referrals,
        "total": len(y_true),
    }


if __name__ == "__main__":
    demo = sample_demo_beats(per_class=2)
    print(f"Sampled {len(demo['X'])} beats across classes: "
          f"{dict(zip(*np.unique(demo['y'], return_counts=True)))}")
    print("Records used:", list(demo["record_ids"]))