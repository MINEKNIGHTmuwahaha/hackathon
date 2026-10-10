"""
src/demo_run.py
---------------
One-command stage demo. The classifier auto-extracts features from raw beats.
"""

import sys
import time
from pathlib import Path
import numpy as np

from .demo import sample_demo_beats
from .logger import get_logger


log = get_logger("sentinel.demo_run")

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def banner(msg: str):
    print(f"\n{BOLD}{CYAN}{'=' * 72}{RESET}")
    print(f"{BOLD}{CYAN}  {msg}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 72}{RESET}\n")


def load_model():
    p = Path("model.pkl")
    if not p.exists():
        return None
    try:
        import joblib
        from .train import TwoStageClassifier
        clf = TwoStageClassifier()
        clf.load(str(p))
        log.info(f"loaded two-stage model from {p}")
        return clf
    except Exception as e:
        log.warning(f"could not load model: {e}")
        return None


def main() -> int:
    t0 = time.perf_counter()
    banner("Sentinel ECG — Live Demo")

    print(f"{BOLD}[1/4] Sampling beats...{RESET}")
    demo = sample_demo_beats(per_class=1, seed=42)
    print(f"      got {len(demo['X'])} beats across "
          f"{len(set(demo['y'].tolist()))} classes")

    print(f"\n{BOLD}[2/4] Loading model...{RESET}")
    model = load_model()
    if model is None:
        print(f"      {YELLOW}no model.pkl found — showing ground truth only{RESET}")
    else:
        print(f"      {GREEN}model loaded ✓{RESET}")

    print(f"\n{BOLD}[3/4] Running inference...{RESET}\n")
    print(f"{'#':>3}  {'record':>7}  {'true':>4}  {'pred':>4}  "
          f"{'conf':>5}  {'decision':>22}")
    print("-" * 74)

    referral_count = 0
    for i, (beat, y_true, rid) in enumerate(zip(demo["X"],
                                                demo["y"],
                                                demo["record_ids"])):
        beat_batch = beat.reshape(1, -1).astype(np.float32)

        if model is not None:
            try:
                y_pred_arr, conf_arr = model.predict(beat_batch)
                y_pred = str(y_pred_arr[0])
                conf = float(conf_arr[0])
            except Exception as e:
                log.warning(f"predict failed: {e}")
                y_pred, conf = "?", 0.0
        else:
            y_pred, conf = y_true, 1.0

        needs_referral = (conf < 0.70) or (y_pred in ("V", "S", "F", "Q"))
        if needs_referral:
            referral_count += 1
            color = RED
            decision = "REFER TO CARDIOLOGIST"
        else:
            color = GREEN
            decision = "normal"

        print(f"{i+1:>3}  {rid:>7}  {y_true:>4}  {y_pred:>4}  "
              f"{conf:>5.2f}  {color}{decision:>22}{RESET}")

    print(f"\n{BOLD}[4/4] Summary{RESET}")
    total = len(demo["X"])
    elapsed = time.perf_counter() - t0
    print(f"      beats processed:     {total}")
    print(f"      referrals triggered: {referral_count}/{total}")
    print(f"      total time:          {elapsed:.2f}s "
          f"({1000 * elapsed / total:.1f} ms/beat)")

    banner("✅ Demo complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())