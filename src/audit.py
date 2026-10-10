"""
src/audit.py
------------
Dataset quality audit — class balance, per-record stats, imbalance,
outliers. Produces docs/dataset_report.json.
"""

import json
from pathlib import Path
import numpy as np

from .constants import DOCS_DIR
from .logger import get_logger


log = get_logger("sentinel.audit")


def audit(X: np.ndarray, y: np.ndarray, record_ids: np.ndarray,
          out_path: str = "docs/dataset_report.json") -> dict:
    classes, counts = np.unique(y, return_counts=True)
    class_counts = {str(c): int(n) for c, n in zip(classes, counts)}

    # Per-record stats
    per_record = {}
    for r in np.unique(record_ids):
        mask = record_ids == r
        per_record[str(r)] = {
            "n_beats": int(mask.sum()),
            "mean": float(X[mask].mean()),
            "std": float(X[mask].std()),
        }

    minority = min(counts)
    majority = max(counts)

    report = {
        "n_beats": int(X.shape[0]),
        "n_features": int(X.shape[1]) if X.ndim > 1 else 1,
        "n_records": int(len(np.unique(record_ids))),
        "class_counts": class_counts,
        "imbalance_ratio": round(majority / minority, 2) if minority else None,
        "signal_stats": {
            "mean": float(X.mean()),
            "std": float(X.std()),
            "min": float(X.min()),
            "max": float(X.max()),
        },
        "per_record": per_record,
    }

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2))
    log.info(f"Audit written to {p}")
    return report


if __name__ == "__main__":
    d = np.load("data/processed.npz", allow_pickle=True)
    r = audit(d["beats"], d["labels"], d["record_ids"])
    print(json.dumps({k: r[k] for k in
                      ("n_beats", "n_records", "class_counts", "imbalance_ratio")},
                     indent=2))