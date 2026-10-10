"""
src/cli.py
----------
Command-line interface. Run as:

    python -m src.cli build-dataset
    python -m src.cli audit
    python -m src.cli split
    python -m src.cli visualize
    python -m src.cli report
"""

import argparse
import json
import numpy as np

from .constants import MITDB_DIR, DATA_DIR
from .data_loader import list_available_records, validate_dataset
from .dataset import build_dataset, build_splits
from .audit import audit
from .visualize import (
    plot_class_distribution, plot_confidence_histogram,
)
from .report import save_metrics_json, save_summary_markdown
from .logger import get_logger


log = get_logger("sentinel.cli")


def cmd_build(args):
    records = list_available_records()
    log.info(f"found {len(records)} records in {MITDB_DIR}")
    data = build_dataset(records, cache_path=args.cache)
    log.info(f"X={data['X'].shape}, y={data['y'].shape}")
    unique, counts = np.unique(data["y"], return_counts=True)
    print("class counts:", dict(zip(unique.tolist(), counts.tolist())))


def cmd_audit(args):
    d = np.load(args.cache, allow_pickle=True)
    rep = audit(d["beats"], d["labels"], d["record_ids"])
    print(json.dumps({k: rep[k] for k in
                      ("n_beats", "n_records", "class_counts",
                       "imbalance_ratio")}, indent=2))


def cmd_split(args):
    d = np.load(args.cache, allow_pickle=True)
    splits = build_splits(d["beats"], d["labels"], d["record_ids"])
    for name, (X, y) in splits.items():
        print(f"{name}: X={X.shape}, y={y.shape}")


def cmd_visualize(args):
    d = np.load(args.cache, allow_pickle=True)
    plot_class_distribution(d["labels"])


def cmd_validate(args):
    rep = validate_dataset()
    print(json.dumps(rep, indent=2))
def cmd_demo(args):
    from .demo import sample_demo_beats
    demo = sample_demo_beats(per_class=args.per_class)
    print(f"Sampled {len(demo['X'])} beats")
    print("Classes:", dict(zip(*np.unique(demo["y"], return_counts=True))))
    print("Records:", list(demo["record_ids"]))

def main():
    p = argparse.ArgumentParser(prog="sentinel")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-dataset")
    b.add_argument("--cache", default="data/processed.npz")
    b.set_defaults(func=cmd_build)

    a = sub.add_parser("audit")
    a.add_argument("--cache", default="data/processed.npz")
    a.set_defaults(func=cmd_audit)

    s = sub.add_parser("split")
    s.add_argument("--cache", default="data/processed.npz")
    s.set_defaults(func=cmd_split)

    v = sub.add_parser("visualize")
    v.add_argument("--cache", default="data/processed.npz")
    v.set_defaults(func=cmd_visualize)

    v2 = sub.add_parser("validate")
    v2.set_defaults(func=cmd_validate)

    d = sub.add_parser("demo")
    d.add_argument("--per-class", type=int, default=1)
    d.set_defaults(func=cmd_demo)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()