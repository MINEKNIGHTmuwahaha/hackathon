"""Smoke tests for src/preprocess.py"""
import numpy as np
from data_loader import load_record, load_record_multilead
from preprocess import (
    bandpass_filter, remove_baseline, segment_beats, normalize_beat,
)


def test_bandpass_preserves_shape():
    signal, _, _ = load_record("100")
    out = bandpass_filter(signal)
    assert out.shape == signal.shape


def test_baseline_removal():
    signal, _, _ = load_record("100")
    out = remove_baseline(signal)
    assert out.shape == signal.shape


def test_segment_beats_shape():
    signal, r_peaks, _ = load_record("100")
    clean = remove_baseline(bandpass_filter(signal))
    beats = segment_beats(clean, r_peaks)
    assert beats.ndim == 2
    assert beats.shape[1] == 210


def test_normalize_beat_zscores():
    signal, r_peaks, _ = load_record("100")
    clean = remove_baseline(bandpass_filter(signal))
    beats = segment_beats(clean, r_peaks)
    nb = normalize_beat(beats[0])
    assert abs(nb.mean()) < 1e-8
    assert abs(nb.std() - 1.0) < 1e-6


def test_multilead_segment():
    sig2, r_peaks, _ = load_record_multilead("100")
    clean = remove_baseline(bandpass_filter(sig2))
    beats = segment_beats(clean, r_peaks)
    assert beats.ndim == 3
    assert beats.shape[1] == 210
    assert beats.shape[2] == 2