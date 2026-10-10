"""
src/digitizer/learn.py
----------------------
Gemini-teacher / System-1-student distillation.

Every time Gemini analyzes an image, we log:
  (S1's raw signal features) → (Gemini's correction)

Over time, we train a small regressor that learns these corrections
so System 1 improves WITHOUT needing Gemini.
"""

import json
from pathlib import Path
from datetime import datetime
import numpy as np

from ..logger import get_logger


log = get_logger("sentinel.learn")

TRAINING_LOG = Path("data/digitizer_training.jsonl")
MODEL_PATH = Path("data/digitizer_corrector.pkl")


def record_pair(image_path: str, s1_signal: np.ndarray,
                s2_signal: np.ndarray, quality: float):
    """
    Log a (S1 output, Gemini output) pair for distillation.
    """
    if s1_signal is None or s2_signal is None:
        return

    # Align lengths (resample smaller to larger)
    n = max(len(s1_signal), len(s2_signal))
    s1r = np.interp(np.linspace(0, 1, n),
                    np.linspace(0, 1, len(s1_signal)), s1_signal)
    s2r = np.interp(np.linspace(0, 1, n),
                    np.linspace(0, 1, len(s2_signal)), s2_signal)

    # Downsample to keep log compact — 100 samples per signal
    n_down = 100
    s1d = np.interp(np.linspace(0, 1, n_down),
                    np.linspace(0, 1, len(s1r)), s1r)
    s2d = np.interp(np.linspace(0, 1, n_down),
                    np.linspace(0, 1, len(s2r)), s2r)

    entry = {
        "ts": datetime.utcnow().isoformat(),
        "image": str(image_path),
        "quality": float(quality),
        "s1": s1d.tolist(),
        "s2": s2d.tolist(),
        "correction": (s2d - s1d).tolist(),
    }

    TRAINING_LOG.parent.mkdir(parents=True, exist_ok=True)
    with TRAINING_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    log.info(f"logged training pair: {image_path}")


def load_training_data() -> list[dict]:
    if not TRAINING_LOG.exists():
        return []
    return [json.loads(l) for l in
            TRAINING_LOG.read_text().splitlines() if l.strip()]


def train_corrector():
    """
    Train a small regressor that predicts Gemini's correction from
    System 1's raw signal + quality score.

    Input features:  S1 signal (100 dims) + quality (1 dim) = 101
    Output:          correction (100 dims)
    """
    data = load_training_data()
    if len(data) < 10:
        log.warning(f"only {len(data)} pairs — need ≥10 to train")
        return None

    from sklearn.ensemble import GradientBoostingRegressor
    import joblib

    X = []
    Y = []
    for entry in data:
        feats = entry["s1"] + [entry["quality"]]
        X.append(feats)
        Y.append(entry["correction"])

    X = np.array(X)
    Y = np.array(Y)

    log.info(f"training corrector on {X.shape}")

    # Multi-output GBR
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )
    model.fit(X, Y)
    joblib.dump(model, MODEL_PATH)
    log.info(f"corrector saved to {MODEL_PATH}")
    return model


def apply_correction(s1_signal: np.ndarray, quality: float) -> np.ndarray:
    """
    Apply the learned correction to System 1's raw signal.
    Falls back to the original if no trained model exists.
    """
    if not MODEL_PATH.exists():
        return s1_signal

    import joblib
    model = joblib.load(MODEL_PATH)

    # Downsample S1 to match training (100 samples)
    n_down = 100
    s1d = np.interp(np.linspace(0, 1, n_down),
                    np.linspace(0, 1, len(s1_signal)), s1_signal)
    feats = np.concatenate([s1d, [quality]]).reshape(1, -1)
    correction_d = model.predict(feats)[0]

    # Upsample correction back to full length
    corr = np.interp(np.linspace(0, 1, len(s1_signal)),
                     np.linspace(0, 1, n_down), correction_d)

    corrected = s1_signal + corr
    return corrected.astype(np.float64)


if __name__ == "__main__":
    data = load_training_data()
    print(f"Training pairs collected: {len(data)}")
    if len(data) >= 10:
        train_corrector()
        print("Corrector trained.")
    else:
        print("Need at least 10 pairs. Use the app to analyze more photos.")