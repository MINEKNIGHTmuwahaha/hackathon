"""
scripts/make_ecg_image.py
-------------------------
Render a realistic ECG printout (with red grid) from a MIT-BIH record.
Outputs a JPEG that looks like a real clinical ECG photo.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path so `src` is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_record
from src.preprocess import bandpass_filter, remove_baseline


def render_ecg_image(record_name: str = "208",
                     duration_s: float = 10.0,
                     out_path: str = "data/samples/test_ecg.jpg"):
    """
    Render a fake ECG printout image.

    duration_s: how many seconds to show (10s = standard ECG strip)
    """
    # Load
    signal, r_peaks, _ = load_record(record_name)
    signal = bandpass_filter(signal)
    signal = remove_baseline(signal)

    # Take first `duration_s` seconds at 360 Hz
    fs = 360
    n = int(duration_s * fs)
    seg = signal[:n]
    t = np.arange(n) / fs

    # ---- Render on ECG paper ----
    fig, ax = plt.subplots(figsize=(14, 5))

    # Paper background (light pink like real ECG paper)
    fig.patch.set_facecolor("#fff5f5")
    ax.set_facecolor("#fff5f5")

    # Minor grid: every 0.04s (1mm), 0.1 mV
    # Major grid: every 0.2s (5mm), 0.5 mV
    ax.set_xticks(np.arange(0, duration_s + 0.01, 0.2), minor=False)
    ax.set_xticks(np.arange(0, duration_s + 0.01, 0.04), minor=True)
    ax.set_yticks(np.arange(-2.0, 2.51, 0.5), minor=False)
    ax.set_yticks(np.arange(-2.0, 2.51, 0.1), minor=True)

    # Major grid = light red
    ax.grid(which="major", color="#ff9999", linewidth=0.6, alpha=0.7)
    # Minor grid = lighter pink
    ax.grid(which="minor", color="#ffcccc", linewidth=0.3, alpha=0.6)

    # ECG trace (black)
    ax.plot(t, seg, color="black", linewidth=1.2)

    # Y-limits
    ax.set_ylim(-1.5, 2.0)
    ax.set_xlim(0, duration_s)

    # Labels like real ECG paper
    ax.set_title(f"ECG — Record {record_name} — 25 mm/s, 10 mm/mV",
                 fontsize=10, color="#333")
    ax.set_xlabel("Time (seconds)", fontsize=9)
    ax.set_ylabel("mV", fontsize=9)

    # Hide axis ticks (real ECG paper has no numbered axis)
    ax.tick_params(axis="both", labelsize=8, colors="#666")

    fig.tight_layout()

    # Save
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, facecolor="#fff5f5")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    # Use only records we KNOW exist in MIT-BIH
    render_ecg_image("208", 10.0, "data/samples/test_pvc.jpg")
    render_ecg_image("100", 10.0, "data/samples/test_normal.jpg")
    render_ecg_image("232", 10.0, "data/samples/test_bradycardia.jpg")
    render_ecg_image("223", 10.0, "data/samples/test_supraventricular.jpg")
    print("done — 4 test images created")