"""
src/smoke.py
------------
End-to-end smoke test. Runs the full Person A pipeline in one command:

    python -m src.smoke

Outputs:
    - data/processed.npz
    - data/splits/{train,val,test}.txt
    - docs/dataset_report.json
    - docs/class_dist.png
    - Prints a summary table
"""

import sys
import time
from pathlib import Path
import numpy as np

from .constants import DATA_DIR, DOCS_DIR, MITDB_DIR
from .data_loader import list_available_records, validate_dataset
from .dataset import build_dataset, build_splits
from .audit import audit
from .visualize import plot_class_distribution
from .logger import get_logger


log = get_logger("sentinel.smoke")


def _hdr(msg: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {msg}")
    print("=" * 72)


def main() -> int:
    t_start = time.perf_counter()
    _hdr("Sentinel ECG — End-to-End Smoke Test")

    # 1. Discover records
    records = list_available_records()
    print(f"[1/6] Records found in {MITDB_DIR}: {len(records)}")
    if not records:
        print("  ✗ No records — put MIT-BIH .dat/.hea files into data/mitdb/")
        return 1

    # 2. Validate
    val = validate_dataset()
    print(f"[2/6] Validation: {val['records_valid']}/{val['records_found']} valid")
    if val["warnings"]:
        for w in val["warnings"][:5]:
            print(f"      ⚠ {w}")

    # 3. Build dataset (cached)
    cache = DATA_DIR / "processed.npz"
    if cache.exists():
        print(f"[3/6] Loading cached dataset: {cache}")
        d = np.load(cache, allow_pickle=True)
        X, y, ids = d["beats"], d["labels"], d["record_ids"]
    else:
        print("[3/6] Building dataset (first run, may take ~30s)...")
        d = build_dataset(records, cache_path=str(cache))
        X, y, ids = d["X"], d["y"], d["record_ids"]

    print(f"      X={X.shape}, y={y.shape}, records={len(np.unique(ids))}")

    # 4. Audit
    print("[4/6] Running audit...")
    rep = audit(X, y, ids)
    for cls in ["N", "S", "V", "F", "Q"]:
        n = rep["class_counts"].get(cls, 0)
        pct = 100.0 * n / rep["n_beats"]
        print(f"      {cls}: {n:>7,}  ({pct:5.2f}%)")
    print(f"      Imbalance ratio: {rep['imbalance_ratio']}")

    # 5. Splits
    print("[5/6] Building patient-separated splits...")
    splits = build_splits(X, y, ids)
    for name, (Xs, ys) in splits.items():
        print(f"      {name:5s}: X={Xs.shape}, y={ys.shape}")

    # 6. Visualization
    print("[6/6] Generating plots...")
    plot_class_distribution(y)

    elapsed = time.perf_counter() - t_start
    _hdr(f"✅ Smoke test complete in {elapsed:.1f}s")
    print(f"  Cache:  {cache}")
    print(f"  Splits: {DATA_DIR / 'splits'}")
    print(f"  Docs:   {DOCS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())