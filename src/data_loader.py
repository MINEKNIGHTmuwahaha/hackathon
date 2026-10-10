"""
src/data_loader.py
------------------
Loads MIT-BIH Arrhythmia Database records using the `wfdb` library.

Contract (do not change — Person B and Person C depend on this):
    load_record(name)            -> (signal, r_peaks, annotations)
    load_record_multilead(name)  -> (signal_2d, r_peaks, annotations)
    list_available_records()     -> list[str]
    get_record_metadata(name)    -> dict
    load_many(names)             -> (beats, labels, record_ids)
    export_processed(names, out) -> dict
    validate_dataset()           -> dict
    benchmark_loading(name)      -> dict
"""

from pathlib import Path
import time
import numpy as np
import wfdb


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mitdb"


# ------------------------------------------------------------------ #
# Discovery
# ------------------------------------------------------------------ #

def list_available_records() -> list[str]:
    """Scan data/mitdb/ for .hea files and return record names."""
    if not DATA_DIR.exists():
        return []
    return sorted(hea.stem for hea in DATA_DIR.glob("*.hea"))


def get_record_metadata(record_name: str) -> dict:
    """Return fs, n_samples, n_channels, n_beats, lead_names for one record."""
    record_path = str(DATA_DIR / record_name)
    rec = wfdb.rdrecord(record_path, physical=False)
    ann = wfdb.rdann(record_path, "atr")
    return {
        "name": record_name,
        "fs": int(rec.fs),
        "n_samples": int(rec.sig_len),
        "n_channels": int(rec.n_sig),
        "n_beats": int(len(ann.sample)),
        "lead_names": list(rec.sig_name),
    }


# ------------------------------------------------------------------ #
# Loading
# ------------------------------------------------------------------ #

def load_record(record_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one MIT-BIH record — single lead (channel 0).

    Returns
    -------
    signal : (n_samples,) float64
    r_peak_samples : (n_beats,) int64
    beat_annotations : (n_beats,) <U4 (MIT-BIH symbols)
    """
    record_path = str(DATA_DIR / record_name)
    rec = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, "atr")

    signal = rec.p_signal[:, 0].astype(np.float64)
    if np.isnan(signal).any():
        signal = np.nan_to_num(signal, nan=0.0)

    r_peaks = np.array(ann.sample, dtype=np.int64)
    annotations = np.array(ann.symbol, dtype=str)

    return signal, r_peaks, annotations


def load_record_multilead(record_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one MIT-BIH record — BOTH leads.

    Returns
    -------
    signal_2d : (n_samples, n_leads) float64
    r_peak_samples : (n_beats,) int64
    beat_annotations : (n_beats,) <U4
    """
    record_path = str(DATA_DIR / record_name)
    rec = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, "atr")

    signal = rec.p_signal.astype(np.float64)
    signal = np.nan_to_num(signal, nan=0.0)

    r_peaks = np.array(ann.sample, dtype=np.int64)
    annotations = np.array(ann.symbol, dtype=str)

    return signal, r_peaks, annotations


# ------------------------------------------------------------------ #
# Batch loading (for Person B / C)
# ------------------------------------------------------------------ #

def load_many(record_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load + preprocess a list of records in one call.

    Returns
    -------
    beats : (total_beats, 210) float64   — filtered, segmented, normalized
    labels : (total_beats,) <U4          — MIT-BIH annotation symbols
    record_ids : (total_beats,) <U8      — record each beat came from
    """
    from preprocess import (
        bandpass_filter, remove_baseline, segment_beats, normalize_beat
    )

    all_beats, all_labels, all_ids = [], [], []

    for name in record_names:
        try:
            signal, r_peaks, ann = load_record(name)
            filtered = bandpass_filter(signal)
            clean = remove_baseline(filtered)
            beats = segment_beats(clean, r_peaks)

            # segment_beats skips edge beats; align annotations to match
            keep = [
                r for r in r_peaks
                if r - 90 >= 0 and r + 120 <= len(signal)
            ]
            ann_kept = ann[:len(keep)]

            beats = np.stack([normalize_beat(b) for b in beats])

            all_beats.append(beats)
            all_labels.append(ann_kept)
            all_ids.append(np.array([name] * len(beats)))

        except Exception as e:
            print(f"[skip] {name}: {e}")
            continue

    if not all_beats:
        return (
            np.empty((0, 210)),
            np.empty((0,), dtype=str),
            np.empty((0,), dtype=str),
        )

    return (
        np.concatenate(all_beats, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_ids, axis=0),
    )


def export_processed(record_names: list[str],
                     out_path: str = "data/processed.npz") -> dict:
    """
    Pre-process a list of records and save a single .npz file.

    Person B can then do:
        d = np.load("data/processed.npz", allow_pickle=True)
        X, y = d["beats"], d["labels"]
    """
    X, y, ids = load_many(record_names)
    if X.shape[0] == 0:
        return {"saved": False, "reason": "no beats loaded"}

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        beats=X.astype(np.float32),
        labels=y.astype("U4"),
        record_ids=ids.astype("U8"),
    )

    return {
        "saved": True,
        "path": str(out),
        "n_beats": int(X.shape[0]),
        "n_records": int(len(set(ids))),
        "size_mb": round(out.stat().st_size / 1024 / 1024, 2),
    }


# ------------------------------------------------------------------ #
# Validation + benchmark
# ------------------------------------------------------------------ #

def validate_dataset() -> dict:
    """Quick sanity check across all available records."""
    records = list_available_records()
    valid, warnings = 0, []

    for name in records:
        try:
            meta = get_record_metadata(name)
            if meta["n_samples"] == 0 or meta["n_beats"] == 0:
                warnings.append(f"{name}: empty")
                continue
            if meta["n_channels"] < 1:
                warnings.append(f"{name}: no channels")
                continue
            valid += 1
        except Exception as e:
            warnings.append(f"{name}: {e}")

    return {
        "records_found": len(records),
        "records_valid": valid,
        "warnings": warnings,
    }


def benchmark_loading(record_name: str = "100", runs: int = 3) -> dict:
    """Time load / filter / segment for one record, averaged over `runs`."""
    from preprocess import bandpass_filter, remove_baseline, segment_beats

    times = {"load_ms": [], "filter_ms": [], "segment_ms": []}
    beats = None

    for _ in range(runs):
        t0 = time.perf_counter()
        signal, r_peaks, _ = load_record(record_name)
        t1 = time.perf_counter()
        filtered = bandpass_filter(signal)
        clean = remove_baseline(filtered)
        t2 = time.perf_counter()
        beats = segment_beats(clean, r_peaks)
        t3 = time.perf_counter()

        times["load_ms"].append((t1 - t0) * 1000)
        times["filter_ms"].append((t2 - t1) * 1000)
        times["segment_ms"].append((t3 - t2) * 1000)

    return {
        "record": record_name,
        "n_beats": int(len(beats)) if beats is not None else 0,
        "load_ms": round(float(np.mean(times["load_ms"])), 2),
        "filter_ms": round(float(np.mean(times["filter_ms"])), 2),
        "segment_ms": round(float(np.mean(times["segment_ms"])), 2),
        "total_ms": round(float(np.mean(times["load_ms"]) +
                                np.mean(times["filter_ms"]) +
                                np.mean(times["segment_ms"])), 2),
    }


# ------------------------------------------------------------------ #
# Test block
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    records = list_available_records()
    print(f"Found {len(records)} records")

    if not records:
        print("No records in data/mitdb/")
        raise SystemExit

    # 1. Single record
    signal, r_peaks, annotations = load_record("100")
    print(f"Signal shape: {signal.shape}")
    print(f"Number of beats: {len(r_peaks)}")
    print(f"First 10 annotations: {annotations[:10]}")

    # 2. Metadata
    print(f"Metadata: {get_record_metadata('100')}")

    # 3. Multi-lead
    sig2, rp2, ann2 = load_record_multilead("100")
    print(f"Multi-lead shape: {sig2.shape}")

    # 4. Validation
    val = validate_dataset()
    print(f"Validation: {val['records_valid']}/{val['records_found']} valid")

    # 5. Benchmark
    print(f"Benchmark: {benchmark_loading('100')}")

    # 6. Export (commented out — takes a few seconds; uncomment to run)
    # print("Export:", export_processed(records))