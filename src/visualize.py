"""
src/visualize.py
----------------
All pitch-quality plots. Each returns the saved path.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless (works in scripts + CI)
import matplotlib.pyplot as plt

from .logger import get_logger


log = get_logger("sentinel.viz")


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {path}")
    return path


def plot_signal(signal, r_peaks, out_path="docs/signal.png", title="ECG"):
    fig, ax = plt.subplots(figsize=(12, 3))
    t = np.arange(len(signal)) / 360.0
    ax.plot(t, signal, lw=0.6, color="#1e5eb0")
    if len(r_peaks):
        ax.scatter(r_peaks / 360.0, signal[r_peaks],
                   color="red", s=8, label="R-peaks")
        ax.legend(loc="upper right")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("mV")
    ax.set_title(title)
    return _save(fig, out_path)


def plot_class_distribution(y, out_path="docs/class_dist.png"):
    unique, counts = np.unique(y, return_counts=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(unique, counts, color="#bc8cff")
    ax.set_ylabel("Count")
    ax.set_title("AAMI class distribution")
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, c,
                f"{c:,}", ha="center", va="bottom", fontsize=9)
    return _save(fig, out_path)


def plot_confusion_matrix(cm, classes, out_path="docs/confusion.png",
                          normalize=True):
    if normalize:
        cm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0,
                   vmax=1 if normalize else float(cm.max()))
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix" + (" (normalized)" if normalize else ""))
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = f"{cm[i, j]:.2f}" if normalize else int(cm[i, j])
            ax.text(j, i, val, ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax)
    return _save(fig, out_path)


def plot_feature_importance(importances, names,
                            out_path="docs/feature_importance.png",
                            top_k=20):
    idx = np.argsort(importances)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([names[i] for i in idx][::-1],
            importances[idx][::-1], color="#3fb950")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_k} features")
    return _save(fig, out_path)


def plot_confidence_histogram(confidences, out_path="docs/confidence_hist.png",
                              threshold=0.70):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(confidences, bins=40, color="#58a6ff", edgecolor="white")
    ax.axvline(threshold, color="red", lw=2,
               label=f"refer threshold = {threshold}")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Model confidence distribution")
    ax.legend()
    return _save(fig, out_path)


def plot_ecg_with_prediction(signal, r_peaks, preds, confs,
                             out_path="docs/pred_strip.png",
                             n_beats=10):
    """Strip plot: waveform + colored beat markers by prediction."""
    colors = {"N": "#3fb950", "S": "#d29922", "V": "#f85149",
              "F": "#bc8cff", "Q": "#8b949e"}

    fig, ax = plt.subplots(figsize=(14, 3))
    t = np.arange(len(signal)) / 360.0
    ax.plot(t, signal, lw=0.5, color="#666")

    for i, (r, p, c) in enumerate(zip(r_peaks[:n_beats],
                                      preds[:n_beats],
                                      confs[:n_beats])):
        col = colors.get(p, "black")
        ax.scatter(r / 360.0, signal[r], color=col, s=40,
                   edgecolor="black", zorder=3)
        ax.annotate(f"{p}\n{c:.2f}", (r / 360.0, signal[r]),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=7)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("mV")
    ax.set_title("Beat-level predictions")
    return _save(fig, out_path)