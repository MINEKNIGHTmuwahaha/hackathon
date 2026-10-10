"""
src/digitizer/ensemble.py
-------------------------
Compare System 1 (offline CV) vs System 2 (Gemini Vision).
Report agreement, disagreements, and per-system strengths.
"""

import numpy as np

from ..logger import get_logger


log = get_logger("sentinel.ensemble")


def compare_systems(sys1: dict, sys2: dict) -> dict:
    """
    sys1: output from sys1_cv.run_on_image() → {ok, signal, r_peaks, quality}
    sys2: output from gemini_vision.analyze_ecg_image() → {ok, result: {...}}

    Returns:
        {
            "n_beats_sys1": int,
            "n_beats_sys2": int,
            "count_agreement": bool,
            "per_beat_agreement": [{"idx1": int, "idx2": int, "dt_s": float}],
            "recommendation": "use_sys1" | "use_sys2" | "review_human",
            "reason": str,
        }
    """
    out = {
        "n_beats_sys1": 0,
        "n_beats_sys2": 0,
        "count_agreement": False,
        "per_beat_agreement": [],
        "recommendation": "review_human",
        "reason": "",
    }

    # System 1
    if sys1.get("ok") and sys1.get("r_peaks") is not None:
        rp1 = sys1["r_peaks"]
        fs1 = sys1.get("fs", 360)
        out["n_beats_sys1"] = len(rp1)
    else:
        rp1, fs1 = np.array([]), 360

    # System 2
    if sys2.get("ok"):
        beats2 = sys2["result"].get("beats") or []
        out["n_beats_sys2"] = len(beats2)
        # Gemini gives positions in pixels normally; if not, we can't align
        rp2_positions = [b.get("estimated_position_px") for b in beats2
                         if b.get("estimated_position_px")]
    else:
        beats2 = []
        rp2_positions = []

    # Beat count agreement within ±15%
    n1, n2 = out["n_beats_sys1"], out["n_beats_sys2"]
    if n1 > 0 and n2 > 0:
        ratio = min(n1, n2) / max(n1, n2)
        out["count_agreement"] = ratio >= 0.85

    # Pick recommendation
    q1 = sys1.get("quality", 0.0) if sys1.get("ok") else 0.0

    if not sys1.get("ok") and sys2.get("ok"):
        out["recommendation"] = "use_sys2"
        out["reason"] = "System 1 (offline CV) failed; Gemini succeeded."
    elif sys1.get("ok") and not sys2.get("ok"):
        out["recommendation"] = "use_sys1"
        out["reason"] = "Gemini unavailable; using offline CV result."
    elif sys1.get("ok") and sys2.get("ok"):
        if out["count_agreement"] and q1 > 0.6:
            out["recommendation"] = "use_sys1"
            out["reason"] = "Both systems agree and System 1 quality is high — prefer offline for speed."
        elif not out["count_agreement"]:
            out["recommendation"] = "review_human"
            out["reason"] = f"Systems disagree on beat count ({n1} vs {n2}). Manual review recommended."
        else:
            out["recommendation"] = "use_sys2"
            out["reason"] = "Both agree; Gemini has richer morphology analysis."
    else:
        out["recommendation"] = "review_human"
        out["reason"] = "Both systems failed or are unavailable."

    return out