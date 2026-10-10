"""
app.py — Sentinel ECG Streamlit demo.
Run:  python -m streamlit run app.py
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from src.data_loader import list_available_records, load_record
from src.preprocess import (
    bandpass_filter, remove_baseline, segment_beats, normalize_beat,
)
from src.features import extract_features
from src.model import ArrhythmiaClassifier


st.set_page_config(page_title="Sentinel ECG", layout="wide")

DANGER = {"V", "S", "F", "Q"}
THRESHOLD = 0.70


@st.cache_resource
def load_model():
    p = Path("model.pkl")
    if not p.exists():
        return None
    from src.train import TwoStageClassifier
    clf = TwoStageClassifier()
    clf.load(str(p))
    return clf


@st.cache_data
def load_metrics():
    p = Path("model_metrics.json")
    if not p.exists():
        return None
    return json.loads(p.read_text())


def predict_record(clf, record_name):
    signal, r_peaks, _ = load_record(record_name)
    filtered = bandpass_filter(signal)
    clean = remove_baseline(filtered)
    beats = segment_beats(clean, r_peaks)

    keep = [i for i, r in enumerate(r_peaks)
            if r - 90 >= 0 and r + 120 <= len(signal)]
    beats = beats[:len(keep)]
    rp = r_peaks[keep]

    beats_norm = np.stack([normalize_beat(b) for b in beats])
    X = extract_features(beats_norm, rp)

    # Apply the SAME feature normalization used during training
    if getattr(clf, "feat_stats", None) is not None:
        X = (X - clf.feat_stats["mean"]) / clf.feat_stats["std"]

    labels, confs = clf.predict(X)
    return signal, rp, beats_norm, labels, confs


st.title("🫀 Project Sentinel — ECG Arrhythmia Triage")
st.caption("Offline AI triage support for rural clinics · Thiraj.io · AICON'26")

clf = load_model()
if clf is None:
    st.error("⚠️ No `model.pkl` found. Train one first: `python -m src.train`")
    st.stop()

records = list_available_records()
if not records:
    st.error("No MIT-BIH records in `data/mitdb/`")
    st.stop()

with st.sidebar:
    st.header("Input")
    record = st.selectbox("Pick a record", records, index=0)
    n_show = st.slider("Beats to display", 5, 40, 15)
    st.divider()
    st.write(f"**Refer threshold:** {THRESHOLD}")
    st.write(f"**Dangerous classes:** V, S, F, Q")

signal, rp, beats, preds, confs = predict_record(clf, record)

n_beats = len(preds)
n_refer = sum(1 for p, c in zip(preds, confs)
              if c < THRESHOLD or p in DANGER)
refer_pct = 100 * n_refer / max(n_beats, 1)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Record", record)
col2.metric("Beats", n_beats)
col3.metric("Referrals", f"{n_refer} ({refer_pct:.1f}%)")
col4.metric("Mean confidence", f"{confs.mean():.2%}")

fig, ax = plt.subplots(figsize=(12, 3))
t = np.arange(len(signal)) / 360.0
ax.plot(t, signal, lw=0.5, color="#1e5eb0")
colors = {"N": "#3fb950", "S": "#d29922", "V": "#f85149",
          "F": "#bc8cff", "Q": "#8b949e"}
for i, (r, p) in enumerate(zip(rp[:n_show], preds[:n_show])):
    ax.scatter(r / 360.0, signal[r], color=colors.get(p, "black"),
               s=40, edgecolor="black", zorder=3)
    ax.annotate(p, (r / 360.0, signal[r]),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=8)
ax.set_xlabel("Time (s)")
ax.set_ylabel("mV")
ax.set_title(f"ECG waveform — record {record}")
st.pyplot(fig)

worst_idx = int(np.argmin(confs))
worst_pred = preds[worst_idx]
worst_conf = confs[worst_idx]

st.divider()
if worst_conf < THRESHOLD or worst_pred in DANGER:
    st.error(
        f"⚠️ **REFER TO CARDIOLOGIST** — "
        f"highest-risk beat: {worst_pred} at {worst_conf:.2%} confidence"
    )
else:
    st.success("✓ All beats within normal range — routine monitoring")

st.divider()
st.subheader("Beat-level predictions")
rows = []
for i in range(min(n_show, n_beats)):
    p, c = preds[i], confs[i]
    need = c < THRESHOLD or p in DANGER
    rows.append({
        "Beat": i + 1,
        "Predicted": p,
        "Confidence": f"{c:.2%}",
        "Decision": "REFER" if need else "normal",
    })
st.dataframe(rows, use_container_width=True)

metrics = load_metrics()
if metrics:
    st.divider()
    st.subheader("Test-set metrics (patient-separated)")
    test = metrics.get("test", {})
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Accuracy", f"{test.get('accuracy', 0):.2%}")
        st.metric("Referral rate", f"{test.get('referral_rate', 0):.2%}")
    with c2:
        for cls, recall in (test.get("per_class_recall") or {}).items():
            st.metric(f"Recall {cls}", f"{recall:.2%}")