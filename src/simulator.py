"""
src/simulator.py
----------------
Simulates a live ECG stream by yielding beats one at a time.
Used by the Streamlit demo to make predictions appear live.

Usage:
    from src.simulator import ECGStream
    stream = ECGStream()
    for beat, meta in stream:
        predict(beat)
"""

import time
import numpy as np

from .data_loader import load_record
from .preprocess import (
    bandpass_filter, remove_baseline, segment_beats, normalize_beat,
)
from .constants import AAMI_MAP
from .logger import get_logger


log = get_logger("sentinel.simulator")


class ECGStream:
    """
    Stream beats from one or more MIT-BIH records, one per interval.

    Args:
        records:      list of record names, e.g. ["208", "232"]
        interval_s:   seconds between beats (0.1 = fast, 1.0 = real-time)
        loop:         if True, restart from the beginning after last beat
        seed:         random seed for record shuffling
    """

    def __init__(self, records=None, interval_s=0.5, loop=True, seed=0):
        self.records = records or ["208", "232", "200", "100"]
        self.interval_s = interval_s
        self.loop = loop
        rng = np.random.default_rng(seed)
        rng.shuffle(self.records)

        self._buffer = []           # list of (signal, r_peaks, annotations, name)
        self._cursor = 0            # which record we're on
        self._beat_idx = 0          # which beat within the current record

        self._load_record(self.records[0])

    def _load_record(self, name: str):
        log.info(f"loading record {name}")
        signal, r_peaks, ann = load_record(name)
        filtered = bandpass_filter(signal)
        clean = remove_baseline(filtered)
        beats = segment_beats(clean, r_peaks)
        beats = np.stack([normalize_beat(b) for b in beats])

        # Keep only beats whose annotations are valid AAMI
        kept_beats, kept_symbols = [], []
        valid_symbols = set()
        for symbols in AAMI_MAP.values():
            valid_symbols.update(symbols)

        for i, sym in enumerate(ann[:len(beats)]):
            if sym in valid_symbols:
                kept_beats.append(beats[i])
                kept_symbols.append(sym)

        self._buffer = (np.array(kept_beats, dtype=np.float32),
                        kept_symbols,
                        name)
        self._beat_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        beats, symbols, name = self._buffer

        if self._beat_idx >= len(beats):
            # move to next record
            self._cursor += 1
            if self._cursor >= len(self.records):
                if not self.loop:
                    raise StopIteration
                self._cursor = 0
            self._load_record(self.records[self._cursor])
            beats, symbols, name = self._buffer

        beat = beats[self._beat_idx]
        symbol = symbols[self._beat_idx]
        self._beat_idx += 1

        meta = {
            "record": name,
            "beat_index": self._beat_idx,
            "true_symbol": symbol,
        }

        time.sleep(self.interval_s)
        return beat, meta


if __name__ == "__main__":
    stream = ECGStream(records=["100"], interval_s=0.05, loop=False)
    for i, (beat, meta) in enumerate(stream):
        print(f"beat {i}: {meta['record']}/{meta['true_symbol']} shape={beat.shape}")
        if i >= 4:
            break