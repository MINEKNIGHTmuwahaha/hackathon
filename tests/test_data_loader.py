"""Smoke tests for src/data_loader.py"""
import numpy as np
from data_loader import (
    list_available_records, load_record, load_record_multilead,
    get_record_metadata, validate_dataset,
)


def test_records_present():
    assert len(list_available_records()) > 0


def test_load_record_shapes():
    signal, r_peaks, ann = load_record("100")
    assert signal.ndim == 1
    assert r_peaks.ndim == 1
    assert len(r_peaks) == len(ann)
    assert signal.dtype == np.float64


def test_multilead_shape():
    sig2, _, _ = load_record_multilead("100")
    assert sig2.ndim == 2
    assert sig2.shape[1] == 2


def test_metadata_fields():
    meta = get_record_metadata("100")
    for k in ("name", "fs", "n_samples", "n_channels", "n_beats"):
        assert k in meta
    assert meta["fs"] == 360


def test_validate():
    rep = validate_dataset()
    assert rep["records_valid"] > 0
    assert rep["records_found"] >= rep["records_valid"]