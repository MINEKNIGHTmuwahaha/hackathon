"""
src/digitizer/sys1_cv.py
------------------------
System 1: Offline OpenCV-based ECG photo digitizer.

Takes a photo of ECG paper and outputs a digital signal estimate.
Runs entirely offline, no API calls.
"""

from pathlib import Path
import json
import numpy as np

from ..logger import get_logger


log = get_logger("sentinel.sys1")

# Tunable parameters — updated by feedback loop
DEFAULT_PARAMS = {
    "grid_red_min": 150,
    "trace_darkness_max": 120,
    "min_trace_pixels_per_col": 2,
    "y_scale_factor": 1.0,
    "noise_floor": 0.15,
}

PARAMS_PATH = Path("data/digitizer_params.json")


def load_params() -> dict:
    if PARAMS_PATH.exists():
        try:
            return {**DEFAULT_PARAMS, **json.loads(PARAMS_PATH.read_text())}
        except Exception:
            pass
    return dict(DEFAULT_PARAMS)


def save_params(params: dict):
    PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PARAMS_PATH.write_text(json.dumps(params, indent=2))


def digitize(image_path: str, out_signal_path: str | None = None) -> dict:
    """
    Photo → digital signal estimate.

    Returns:
        {
            "ok": bool,
            "error": str | None,
            "signal": np.ndarray | None,        # shape (N,)
            "fs": int,                          # sampling rate
            "duration_s": float,
            "r_peaks": np.ndarray | None,       # estimated R-peak positions
            "quality": float,                   # 0-1 confidence
            "params_used": dict,
        }
    """
    try:
        import cv2
    except ImportError:
        return {"ok": False, "error": "opencv-python not installed. Run: python -m pip install opencv-python",
                "signal": None, "r_peaks": None, "quality": 0.0}

    p = Path(image_path)
    if not p.exists():
        return {"ok": False, "error": f"File not found: {p}",
                "signal": None, "r_peaks": None, "quality": 0.0}

    params = load_params()

    # ---- 1. Load image ----
    img = cv2.imread(str(p))
    if img is None:
        return {"ok": False, "error": "Could not read image",
                "signal": None, "r_peaks": None, "quality": 0.0}

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---- 2. Remove grid (red channel from BGR) ----
    b, g, r = cv2.split(img)
    # ECG grid is typically red/pink → high R, lower G/B
    # The trace is black → low R, low G, low B
    # Use red channel: grid has high values, trace has low values
    red = r
    # Trace mask: where red is dark
    trace_mask = (red < params["trace_darkness_max"]).astype(np.uint8) * 255

    # ---- 3. Extract trace: per-column, take median Y of trace pixels ----
    signal_y = np.full(w, np.nan, dtype=np.float64)
    trace_counts = np.zeros(w, dtype=np.int32)

    for x in range(w):
        col = trace_mask[:, x]
        ys = np.where(col > 0)[0]
        trace_counts[x] = len(ys)
        if len(ys) >= params["min_trace_pixels_per_col"]:
            # Take the median Y — robust to noise
            signal_y[x] = np.median(ys)

    # ---- 4. Interpolate gaps ----
    valid = ~np.isnan(signal_y)
    if valid.sum() < 10:
        return {"ok": False, "error": "Too few valid columns — trace not detected",
                "signal": None, "r_peaks": None, "quality": 0.0}

    x_valid = np.where(valid)[0]
    y_valid = signal_y[valid]
    signal_y = np.interp(np.arange(w), x_valid, y_valid)

    # ---- 5. Invert Y (image Y grows down; ECG mV grows up) ----
    signal = -signal_y

    # ---- 6. Remove baseline (moving average) ----
    from scipy.ndimage import uniform_filter1d
    window = max(5, w // 50)
    baseline = uniform_filter1d(signal, size=window)
    signal = signal - baseline

    # ---- 7. Estimate sampling rate ----
    # Assume standard ECG paper: 25 mm/s. Need to know mm-per-pixel.
    # Conservative assumption: image width = ~10 seconds (standard ECG strip).
    # This gets refined by the feedback loop.
    duration_s = 10.0
    fs = 360
    n_out = int(duration_s * fs)

    # Resample x-axis to n_out samples
    x_orig = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, n_out)
    signal = np.interp(x_new, x_orig, signal)

    # ---- 8. Apply y-scale factor (learned) ----
    signal = signal * params["y_scale_factor"]

    # Normalize amplitude to a reasonable range (~±1.5 mV)
    peak = np.abs(signal).max()
    if peak > 0.01:
        signal = signal / peak * 1.5

    # ---- 9. Estimate quality score ----
    # Quality = fraction of columns with a valid trace reading
    valid_frac = valid.sum() / max(w, 1)
    # Penalize if the trace is extremely noisy
    high_freq = np.abs(np.diff(signal)).mean()
    noise_penalty = min(high_freq / 0.1, 1.0)
    quality = float(np.clip(valid_frac - 0.3 * noise_penalty, 0, 1))

    # ---- 10. Detect R-peaks for comparison ----
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(signal, distance=fs // 5, height=np.std(signal) * 0.5)

    return {
        "ok": True,
        "error": None,
        "signal": signal,
        "fs": fs,
        "duration_s": duration_s,
        "r_peaks": peaks,
        "quality": round(quality, 3),
        "params_used": params,
    }


def run_on_image(image_path: str) -> dict:
    """Wraps digitize() with error handling for the UI."""
    try:
        return digitize(image_path)
    except Exception as e:
        log.error(f"System 1 failed: {e}")
        return {"ok": False, "error": str(e),
                "signal": None, "r_peaks": None, "quality": 0.0}
    # Optional: apply learned Gemini corrections
    from .learn import apply_correction
    if result["ok"] and result["signal"] is not None:
        result["signal_raw"] = result["signal"].copy()
        result["signal"] = apply_correction(result["signal"],
                                             result["quality"])
    return result

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/samples/test_pvc.jpg"
    result = run_on_image(path)
    if result["ok"]:
        print(f"✓ {path}")
        print(f"  signal: {result['signal'].shape}")
        print(f"  r-peaks: {len(result['r_peaks'])}")
        print(f"  quality: {result['quality']}")
    else:
        print(f"✗ {result['error']}")