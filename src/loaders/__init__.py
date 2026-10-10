"""
src/loaders/__init__.py
-----------------------
Registry of dataset loaders. Each loader exposes:
    list_records()               -> list[str]
    load_record(name)            -> (beats, labels, r_peaks, record_id)
    dataset_id                   -> str
"""

from . import mitdb
from . import svdb


DATASETS = {
    "mitdb": mitdb,
    "svdb":  svdb,
}


def get_loader(name: str):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. "
                         f"Available: {list(DATASETS.keys())}")
    return DATASETS[name]


def list_all_records() -> dict:
    """Return {dataset_name: [record_names]}."""
    return {name: mod.list_records() for name, mod in DATASETS.items()}