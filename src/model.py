"""
src/model.py
------------
AAMI arrhythmia classifier wrappers (feature-aware, normalization-aware).
"""

from pathlib import Path
import numpy as np

from .constants import AAMI_MAP, AAMI_CLASSES
from .logger import get_logger


log = get_logger("sentinel.model")
BEAT_WINDOW = 210


def map_mitbih_to_aami(symbols) -> np.ndarray:
    sym_to_aami = {}
    for cls, syms in AAMI_MAP.items():
        for s in syms:
            sym_to_aami[s] = cls
    out = [sym_to_aami[str(s)] for s in symbols if str(s) in sym_to_aami]
    return np.array(out, dtype="U1")


class ArrhythmiaClassifier:
    """HistGradientBoosting with auto feature extraction."""

    def __init__(self, n_estimators: int = 300, random_state: int = 42):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.model = HistGradientBoostingClassifier(
            max_iter=n_estimators,
            learning_rate=0.08,
            max_depth=8,
            random_state=random_state,
        )
        self.fitted = False

    @staticmethod
    def _prepare(X: np.ndarray, r_peaks=None) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError(f"expected 2D array, got {X.shape}")
        if X.shape[1] == BEAT_WINDOW:
            from .features import extract_features
            if r_peaks is None:
                r_peaks = np.full(X.shape[0], BEAT_WINDOW // 2, dtype=np.int64)
            return extract_features(X, r_peaks)
        return X

    def fit(self, X: np.ndarray, y: np.ndarray, r_peaks=None) -> None:
        Xf = self._prepare(X, r_peaks)
        from collections import Counter
        counts = Counter(y.tolist())
        n_total = len(y)
        n_classes = len(counts)
        weights = np.array(
            [n_total / (n_classes * counts[label]) for label in y],
            dtype=np.float64,
        )
        log.info(f"fitting on X={Xf.shape}, y={y.shape}")
        self.model.fit(Xf, y, sample_weight=weights)
        self.fitted = True

    def predict(self, X: np.ndarray, r_peaks=None):
        if not self.fitted:
            raise RuntimeError("model not fitted")
        Xf = self._prepare(X, r_peaks)
        proba = self.model.predict_proba(Xf)
        classes = self.model.classes_
        idx = proba.argmax(axis=1)
        return classes[idx].astype("U1"), proba.max(axis=1).astype(np.float64)

    def feature_importances(self) -> np.ndarray:
        if not self.fitted:
            return np.array([])
        return getattr(self.model, "feature_importances_", np.array([]))

    def save(self, path: str = "model.pkl") -> None:
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        log.info(f"model saved to {path}")

    def load(self, path: str = "model.pkl") -> None:
        import joblib
        self.model = joblib.load(path)
        self.fitted = True
        log.info(f"model loaded from {path}")


class EnsembleClassifier:
    """Soft-voting: HistGB + RF + LogisticRegression."""

    def __init__(self, random_state: int = 42):
        from sklearn.ensemble import (
            HistGradientBoostingClassifier, RandomForestClassifier,
        )
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        self.hgb = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_depth=8,
            random_state=random_state,
        )
        self.rf = RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=random_state, n_jobs=-1,
        )
        self.lr = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
        ])
        self.fitted = False

    @staticmethod
    def _prepare(X, r_peaks=None):
        return ArrhythmiaClassifier._prepare(X, r_peaks)

    def fit(self, X, y, r_peaks=None):
        Xf = self._prepare(X, r_peaks)
        from collections import Counter
        counts = Counter(y.tolist())
        n = len(y)
        k = len(counts)
        w = np.array([n / (k * counts[lbl]) for lbl in y])
        self.hgb.fit(Xf, y, sample_weight=w)
        self.rf.fit(Xf, y)
        self.lr.fit(Xf, y)
        self.fitted = True

    def predict(self, X, r_peaks=None):
        Xf = self._prepare(X, r_peaks)
        hgb_p = self.hgb.predict_proba(Xf)
        rf_p = self.rf.predict_proba(Xf)
        lr_p = self.lr.predict_proba(Xf)
        avg = (hgb_p + rf_p + lr_p) / 3.0
        idx = avg.argmax(axis=1)
        classes = self.hgb.classes_
        return classes[idx].astype("U1"), avg.max(axis=1).astype(np.float64)


def evaluate(y_true, y_pred, y_conf, threshold: float = 0.70) -> dict:
    from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_conf = np.asarray(y_conf)

    labels = list(AAMI_CLASSES)
    acc = float(accuracy_score(y_true, y_pred))
    recalls = recall_score(y_true, y_pred, labels=labels,
                           average=None, zero_division=0)
    per_class_recall = {lbl: float(r) for lbl, r in zip(labels, recalls)}
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    referral_rate = float((y_conf < threshold).mean())

    return {
        "accuracy": round(acc, 4),
        "per_class_recall": {k: round(v, 4) for k, v in per_class_recall.items()},
        "confusion_matrix": cm,
        "class_labels": labels,
        "referral_rate": round(referral_rate, 4),
        "n_samples": int(len(y_true)),
    }