"""
src/config.py
-------------
Load YAML config into a simple dict-like object with attribute access.
"""

from pathlib import Path
import yaml


class Config:
    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, key):
        if key in self._data:
            val = self._data[key]
            return Config(val) if isinstance(val, dict) else val
        raise AttributeError(key)

    def __getitem__(self, key):
        return self._data[key]

    def as_dict(self) -> dict:
        return self._data


def load_config(path: str = "configs/default.yaml") -> Config:
    p = Path(path)
    if not p.exists():
        return Config({})
    with open(p, "r", encoding="utf-8") as f:
        return Config(yaml.safe_load(f) or {})