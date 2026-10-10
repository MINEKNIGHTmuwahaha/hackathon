"""
src/train.py
------------
Two-stage hierarchical classifier with feature normalization.

Stage 1: N vs not-N (balanced)
Stage 2: S/V/F/Q (balanced)

Features are z-scored using training statistics; stats saved in model.pkl.
At inference: if raw beats (210) are passed, features are extracted and
normalized. If features (41) are passed, they are assumed already normalized.
"""

import json
from pathlib import Path
from collections import Counter
import numpy as np

from .loaders import DATASETS
from .splits import make_patient_split
from .features import extract_features, get_feature_names, normalize_features
from .model import ArrhythmiaClassifier, EnsembleClassifier, evaluate
from .logger import get_logger


log = get_logger("sentinel.train")


def _load_beats_from_dataset(dataset_name: str, record_name: str):
    from .loaders import get_loader
    return get_loader(dataset_name).load_record(record_name)


def train_and_pick_best(X_train, y_train, X_val, y_val, log_prefix=""):
    """Try several models, evaluate on val, return best by balanced accuracy."""
    from sklearn.metrics import balanced_accuracy_score

    candidates = {
        "hist_gb":  ArrhythmiaClassifier(n_estimators=300),
        "ensemble": EnsembleClassifier(),
    }

    best_name, best_score, best_clf = None, -1.0, None
    for name, clf in candidates.items():
        log.info(f"{log_prefix} trying {name}")
        try:
            clf.fit(X_train, y_train)
            pred, conf = clf.predict(X_val)
            score = balanced_accuracy_score(y_val, pred)
            log.info(f"{log_prefix} {name}: balanced_acc={score:.4f}")
            if score > best_score:
                best_name, best_score, best_clf = name, score, clf
        except Exception as e:
            log.warning(f"{log_prefix} {name} failed: {e}")

    if best_clf is None:
        raise RuntimeError(f"{log_prefix} all candidates failed")

    log.info(f"{log_prefix} best: {best_name} ({best_score:.4f})")
    return best_clf, best_name


class TwoStageClassifier:
    """N vs not-N → S/V/F/Q, with feature normalization preserved."""

    def __init__(self):
        self.stage1 = None
        self.stage2 = None
        self.feat_stats = None
        self.fitted = False

    def _normalize(self, X):
        if self.feat_stats is None:
            return X
        return (X - self.feat_stats["mean"]) / self.feat_stats["std"]

    def fit(self, X, y, feat_stats=None):
        """
        X is expected to be ALREADY normalized features (n, 41).
        feat_stats are stored for later use during predict on raw beats.
        """
        self.feat_stats = feat_stats

        n = len(X)
        rng = np.random.default_rng(42)
        idx = rng.permutation(n)
        cut = int(0.8 * n)
        tr, va = idx[:cut], idx[cut:]

        # Stage 1: N vs not-N
        y_bin = np.where(y == "N", "N", "X")
        log.info("stage 1: model selection")
        self.stage1, s1_name = train_and_pick_best(
            X[tr], y_bin[tr], X[va], y_bin[va], log_prefix="[s1]"
        )

        # Stage 2: non-N only
        mask = y != "N"
        X2, y2 = X[mask], y[mask]
        n2 = len(X2)
        rng2 = np.random.default_rng(42)
        idx2 = rng2.permutation(n2)
        cut2 = int(0.8 * n2)
        tr2, va2 = idx2[:cut2], idx2[cut2:]

        log.info("stage 2: model selection")
        self.stage2, s2_name = train_and_pick_best(
            X2[tr2], y2[tr2], X2[va2], y2[va2], log_prefix="[s2]"
        )
        log.info(f"stage1={s1_name}, stage2={s2_name}")
        self.fitted = True

    def predict(self, X):
        """
        Accepts either:
          - raw beats  (n, 210): extracts features and applies normalization
          - features   (n, 41):  assumed already normalized (uses as-is)
        """
        # Detect whether X is raw beats or pre-extracted features
        is_raw_beats = (X.ndim == 2 and X.shape[1] == 210)

        if is_raw_beats:
            # Extract features and normalize with saved stats
            X_prepared = ArrhythmiaClassifier._prepare(X)
            X_prepared = self._normalize(X_prepared)
        else:
            # Already features — assume already normalized
            X_prepared = X

        # Stage 1
        pred_bin, conf_bin = self.stage1.predict(X_prepared)

        out_pred = np.empty(len(X_prepared), dtype="U1")
        out_conf = np.zeros(len(X_prepared))

        # N predictions
        n_mask = pred_bin == "N"
        out_pred[n_mask] = "N"
        out_conf[n_mask] = conf_bin[n_mask]

        # Stage 2 on non-N
        x_mask = ~n_mask
        if x_mask.any() and self.stage2 is not None:
            p2, c2 = self.stage2.predict(X_prepared[x_mask])
            out_pred[x_mask] = p2
            out_conf[x_mask] = c2

        return out_pred, out_conf

    def save(self, path="model.pkl"):
        import joblib
        joblib.dump({
            "stage1": self.stage1,
            "stage2": self.stage2,
            "feat_stats": self.feat_stats,
        }, path)
        log.info(f"two-stage model saved to {path}")

    def load(self, path="model.pkl"):
        import joblib
        d = joblib.load(path)
        self.stage1 = d["stage1"]
        self.stage2 = d["stage2"]
        self.feat_stats = d.get("feat_stats")
        self.fitted = True


def train_pipeline(record_names=None, model_path="model.pkl",
                   val_ratio=0.15, test_ratio=0.15, seed=42):
    log.info("=== Multi-dataset training ===")

    all_beats, all_rp, all_labels, all_ids = [], [], [], []

    for ds_name in DATASETS.keys():
        try:
            records = DATASETS[ds_name].list_records()
        except Exception as e:
            log.warning(f"{ds_name} unavailable: {e}")
            continue
        if not records:
            log.info(f"{ds_name}: no records — skipping")
            continue

        log.info(f"{ds_name}: {len(records)} records")
        for name in records:
            try:
                beats, labels, rp, rid = _load_beats_from_dataset(ds_name, name)
                if beats is None:
                    continue
                all_beats.append(beats)
                all_rp.append(rp)
                all_labels.append(labels)
                all_ids.append(np.array([rid] * len(beats), dtype="U32"))
                log.info(f"  {rid}: {len(beats)} beats")
            except Exception as e:
                log.error(f"  {ds_name}/{name}: {e}")

    if not all_beats:
        raise RuntimeError("no beats loaded from any dataset")

    beats = np.concatenate(all_beats, axis=0)
    rpeaks = np.concatenate(all_rp, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    ids = np.concatenate(all_ids, axis=0)

    log.info(f"total beats: {len(beats)}")
    log.info("extracting features")
    X = extract_features(beats, rpeaks)

    # Patient-separated split
    split = make_patient_split(list(ids), 1.0 - val_ratio - test_ratio,
                               val_ratio, test_ratio, seed)
    train_mask = np.isin(ids, split["train"])
    val_mask = np.isin(ids, split["val"])
    test_mask = np.isin(ids, split["test"])

    X_train, y_train = X[train_mask], labels[train_mask]
    X_val, y_val = X[val_mask], labels[val_mask]
    X_test, y_test = X[test_mask], labels[test_mask]

    log.info(f"train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    log.info(f"train counts: {dict(Counter(y_train.tolist()))}")

    # ---- Feature normalization (fit stats on train only) ----
    log.info("normalizing features")
    X_train, feat_stats = normalize_features(X_train)
    X_val, _ = normalize_features(X_val, feat_stats)
    X_test, _ = normalize_features(X_test, feat_stats)

    # ---- Fit two-stage ----
    log.info("fitting two-stage classifier")
    clf = TwoStageClassifier()
    clf.fit(X_train, y_train, feat_stats=feat_stats)

    # ---- Evaluate (X_val/X_test are already normalized) ----
    y_pred_val, conf_val = clf.predict(X_val)
    y_pred_test, conf_test = clf.predict(X_test)

    metrics_val = evaluate(y_val, y_pred_val, conf_val)
    metrics_test = evaluate(y_test, y_pred_test, conf_test)

    clf.save(model_path)

    metrics_out = {
        "val": metrics_val,
        "test": metrics_test,
        "feature_names": get_feature_names(),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "records_train": split["train"],
        "records_val": split["val"],
        "records_test": split["test"],
        "datasets": list(DATASETS.keys()),
    }

    Path("model_metrics.json").write_text(json.dumps(metrics_out, indent=2))
    log.info("metrics saved")

    return metrics_out


if __name__ == "__main__":
    metrics = train_pipeline()
    print(json.dumps(metrics["test"], indent=2))