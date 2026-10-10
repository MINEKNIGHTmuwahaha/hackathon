"""
src/digitizer/feedback.py
-------------------------
Store user feedback on which system was more correct.
Use accumulated feedback to auto-tune System 1 parameters.
"""

from pathlib import Path
from datetime import datetime
import json
import numpy as np

from ..logger import get_logger


log = get_logger("sentinel.feedback")

FEEDBACK_LOG = Path("data/digitizer_feedback.jsonl")
PARAMS_PATH = Path("data/digitizer_params.json")


def record_feedback(image_path: str, sys1_result: dict, sys2_result: dict,
                    winner: str, notes: str = ""):
    """
    winner: "sys1" | "sys2" | "both" | "neither"
    Appends one line to feedback log.
    """
    entry = {
        "ts": datetime.utcnow().isoformat(),
        "image": str(image_path),
        "winner": winner,
        "sys1_ok": sys1_result.get("ok", False),
        "sys1_quality": sys1_result.get("quality", 0.0),
        "sys1_n_peaks": len(sys1_result.get("r_peaks", []) or []),
        "sys2_ok": sys2_result.get("ok", False),
        "sys2_n_beats": len((sys2_result.get("result") or {}).get("beats") or []),
        "notes": notes,
    }
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    log.info(f"feedback logged: {winner}")


def load_feedback() -> list[dict]:
    if not FEEDBACK_LOG.exists():
        return []
    return [json.loads(line) for line in FEEDBACK_LOG.read_text().splitlines() if line.strip()]


def auto_tune_params():
    """
    Read feedback history and update System 1 parameters.
    Called after each feedback entry.
    """
    history = load_feedback()
    if len(history) < 3:
        return None

    # Load current params
    from .sys1_cv import load_params, save_params
    params = load_params()

    # Recent entries (last 10)
    recent = history[-10:]

    # Case 1: Sys1 keeps failing but Gemini succeeds → loosen trace darkness
    sys1_fails = sum(1 for e in recent if not e["sys1_ok"])
    if sys1_fails > len(recent) * 0.5:
        params["trace_darkness_max"] = min(params["trace_darkness_max"] + 10, 180)
        log.info(f"auto-tune: raised trace_darkness_max to {params['trace_darkness_max']}")

    # Case 2: Sys1 too noisy (low quality) → require more trace pixels
    low_quality = sum(1 for e in recent if e["sys1_quality"] < 0.4)
    if low_quality > len(recent) * 0.5:
        params["min_trace_pixels_per_col"] = min(params["min_trace_pixels_per_col"] + 1, 10)
        log.info(f"auto-tune: raised min_trace_pixels to {params['min_trace_pixels_per_col']}")

    # Case 3: Sys1 beat count always off → scale Y-axis
    count_mismatch = sum(
        1 for e in recent
        if e["sys1_n_peaks"] > 0 and abs(e["sys1_n_peaks"] - e["sys2_n_beats"]) > 3
    )
    if count_mismatch > len(recent) * 0.6:
        params["y_scale_factor"] *= 0.95
        log.info(f"auto-tune: adjusted y_scale_factor to {params['y_scale_factor']:.3f}")

    save_params(params)
    return params


if __name__ == "__main__":
    history = load_feedback()
    print(f"{len(history)} feedback entries")
    if history:
        print(json.dumps(history[-3:], indent=2))
    tuned = auto_tune_params()
    if tuned:
        print("\nTuned params:", json.dumps(tuned, indent=2))