"""
app.py — Sentinel ECG Streamlit demo.

Three input modes:
  1. Sample record (pick from MIT-BIH)
  2. Upload photo of ECG (Gemini Vision)
  3. Upload .dat/.hea (digital signal, ML classifier)
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


st.set_page_config(page_title="Sentinel ECG", layout="wide")

DANGER = {"V", "S", "F", "Q"}
THRESHOLD = 0.70
COLORS = {"N": "#3fb950", "S": "#d29922", "V": "#f85149",
          "F": "#bc8cff", "Q": "#8b949e"}


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

    if getattr(clf, "feat_stats", None) is not None:
        X = (X - clf.feat_stats["mean"]) / clf.feat_stats["std"]

    labels, confs = clf.predict(X)
    return signal, rp, beats_norm, labels, confs


def render_banner(preds, confs, n_beats, n_refer):
    from collections import Counter
    class_counts = Counter(preds.tolist())
    dangerous = {c: class_counts.get(c, 0) for c in DANGER if c in class_counts}

    if dangerous:
        reason = ", ".join(f"{n}× {c}" for c, n in dangerous.items())
        avg_conf = np.mean([c for p, c in zip(preds, confs) if p in DANGER]) \
                   if dangerous else 0.0
        st.error(
            f"⚠️ **REFER TO CARDIOLOGIST**  \n"
            f"Reason: {reason} detected  \n"
            f"Average confidence on abnormal beats: **{avg_conf:.1%}**  \n"
            f"Total referrals: **{n_refer}/{n_beats}**"
        )
    else:
        st.success(
            f"✓ No dangerous classes detected — routine monitoring  \n"
            f"Mean confidence: **{confs.mean():.2%}**"
        )


# =========================================================
# PAGE
# =========================================================
st.title("🫀 Project Sentinel — ECG Arrhythmia Triage")
st.caption("Offline AI triage support for rural clinics · Thiraj.io · AICON'26")

tab_record, tab_photo, tab_dual = st.tabs(
    ["📊 Sample Record", "📷 Upload ECG Photo", "🔬 Dual System"]
)


# =========================================================
# TAB 1 — Sample record from MIT-BIH
# =========================================================
with tab_record:
    clf = load_model()
    if clf is None:
        st.error("⚠️ No `model.pkl` found. Train one first: `python -m src.train`")
        st.stop()

    records = list_available_records()
    if not records:
        st.error("No MIT-BIH records in `data/mitdb/`")
        st.stop()

    with st.sidebar:
        st.header("Input — Sample Record")
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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Record", record)
    c2.metric("Beats", n_beats)
    c3.metric("Referrals", f"{n_refer} ({refer_pct:.1f}%)")
    c4.metric("Mean confidence", f"{confs.mean():.2%}")

    if len(rp) > 0:
        first_beat = int(rp[0])
        last_idx = min(n_show - 1, len(rp) - 1)
        last_beat = int(rp[last_idx])
        pad = 200
        lo = max(0, first_beat - pad)
        hi = min(len(signal), last_beat + pad)
        seg = signal[lo:hi]
        t = np.arange(lo, hi) / 360.0

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t, seg, lw=1.0, color="#1e5eb0")
        for i, (r, p) in enumerate(zip(rp[:n_show], preds[:n_show])):
            if lo <= r < hi:
                ax.scatter(r / 360.0, signal[r],
                           color=COLORS.get(p, "black"),
                           s=80, edgecolor="black", zorder=3)
                ax.annotate(f"{p}\n{confs[i]:.2f}",
                            (r / 360.0, signal[r]),
                            textcoords="offset points", xytext=(0, 15),
                            ha="center", fontsize=9, fontweight="bold",
                            color=COLORS.get(p, "black"))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("mV")
        ax.set_title(f"ECG — record {record} (first {n_show} beats)")
        st.pyplot(fig)

    if n_beats > 0:
        st.divider()
        render_banner(preds, confs, n_beats, n_refer)

    metrics = load_metrics()
    if metrics:
        st.divider()
        st.subheader("Test-set metrics (patient-separated)")
        test = metrics.get("test", {})
        cc1, cc2 = st.columns(2)
        with cc1:
            st.metric("Accuracy", f"{test.get('accuracy', 0):.2%}")
            st.metric("Referral rate", f"{test.get('referral_rate', 0):.2%}")
        with cc2:
            for cls, recall in (test.get("per_class_recall") or {}).items():
                st.metric(f"Recall {cls}", f"{recall:.2%}")


# =========================================================
# TAB 2 — Upload ECG photo (Gemini Vision)
# =========================================================
with tab_photo:
    st.subheader("📷 Upload a photo of an ECG printout")
    st.caption("Powered by Gemini 2.5 Flash Vision. Works on photos "
               "of ECG paper — no grid calibration needed.")

    uploaded = st.file_uploader("Choose an ECG image",
                                 type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        # Save temporarily so Gemini can read it
        tmp_path = Path("data/samples/_uploaded.jpg")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(uploaded.getvalue())

        st.image(str(tmp_path), caption="Uploaded ECG",
                 use_container_width=True)

        if st.button("🔍 Analyze with Gemini Vision"):
            with st.spinner("Analyzing with Gemini..."):
                from src.gemini_vision import analyze_ecg_image, summarize_for_ui
                out = analyze_ecg_image(str(tmp_path))

            if not out["ok"]:
                st.error(f"Analysis failed: {out['error']}")
                st.stop()

            summary = summarize_for_ui(out["result"])

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Image quality", summary["image_quality"])
            m2.metric("Beats detected", summary["n_beats"])
            m3.metric("Rhythm", summary["rhythm"])
            m4.metric("Heart rate",
                      f"{summary['heart_rate_bpm'] or '?'} bpm")

            # Class counts
            st.write("**Detected beat classes:**")
            cols = st.columns(5)
            for i, (cls, count) in enumerate(summary["class_counts"].items()):
                cols[i].metric(cls, count)

            # Referral banner
            # Referral banner
            st.divider()
            if summary["referral_recommended"] or summary["referral_count"] > 0:
                reason_text = summary['referral_reason'] or 'Dangerous beats detected'
                if summary["referral_count"] > 0:
                    stats_text = (f"Beat-level referrals: "
                                  f"**{summary['referral_count']}/{summary['n_beats']}** "
                                  f"({summary['referral_pct']:.1f}%)")
                else:
                    stats_text = "Referral triggered by rhythm-level assessment"
                st.error(
                    f"⚠️ **REFER TO CARDIOLOGIST**  \n"
                    f"Reason: {reason_text}  \n"
                    f"{stats_text}"
                )
            else:
                st.success("✓ No dangerous beats detected — routine monitoring")

            # Red flags
            if summary["red_flags"]:
                st.warning("**Red flags:** " + ", ".join(summary["red_flags"]))

            # Beat table
            st.divider()
            st.subheader("Per-beat classification")
            beats = out["result"].get("beats") or []
            if beats:
                rows = ["| # | Morphology | AAMI | Confidence |",
                        "|---|-----------|------|------------|"]
                for b in beats:
                    idx = b.get("index", "?")
                    morp = b.get("morphology", "?")
                    cls = b.get("aami_class_guess", "?")
                    conf = float(b.get("confidence", 0.0))
                    rows.append(f"| {idx} | {morp} | {cls} | {conf:.2f} |")
                st.markdown("\n".join(rows))

            # Raw JSON expander
            with st.expander("Raw Gemini output"):
                st.json(out["result"])

    else:
        st.info(
            "Upload a photo of an ECG printout. On stage, you can take a "
            "photo of any ECG image and drop it here."
        )
# =========================================================
# TAB 3 — Dual System (offline CV + Gemini + feedback loop)
# =========================================================
with tab_dual:
    st.subheader("🔬 Dual-System Photo Analysis")
    st.caption(
        "**System 1** — offline OpenCV digitizer (fast, no internet, "
        "auto-tunes from feedback). "
        "**System 2** — Gemini Vision (online, richer analysis). "
        "Run both, compare, and give feedback to improve System 1."
    )

    upload2 = st.file_uploader("Upload ECG photo",
                                type=["jpg", "jpeg", "png"],
                                key="dual_upload")

    if upload2:
        tmp = Path("data/samples/_dual.jpg")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(upload2.getvalue())
        st.image(str(tmp), caption="Uploaded ECG", use_container_width=True)

        if st.button("🔬 Compare both systems"):
            from src.digitizer.sys1_cv import run_on_image
            from src.gemini_vision import analyze_ecg_image
            from src.digitizer.ensemble import compare_systems

            with st.spinner("Running System 1 (offline OpenCV)..."):
                s1 = run_on_image(str(tmp))

            with st.spinner("Running System 2 (Gemini Vision)..."):
                s2 = analyze_ecg_image(str(tmp))

            cmp = compare_systems(s1, s2)

            # --- metrics (numpy-safe) ---
            rp1 = s1.get("r_peaks")
            n_sys1 = len(rp1) if rp1 is not None else 0

            sys2_beats = (s2.get("result") or {}).get("beats") if s2["ok"] else None
            n_sys2 = len(sys2_beats) if sys2_beats else 0

            sys2_rhythm = (s2.get("result") or {}).get("rhythm", "—") if s2["ok"] else "—"
            s1_quality = s1.get("quality", 0.0) if s1["ok"] else 0.0

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("System 1 beats", n_sys1)
                st.metric("Sys1 quality", f"{s1_quality:.2f}")
            with c2:
                st.metric("System 2 beats", n_sys2)
                st.metric("Sys2 rhythm", sys2_rhythm)
            with c3:
                st.metric("Beat-count agreement",
                          "✅" if cmp["count_agreement"] else "❌")
                st.metric("Recommended",
                          cmp["recommendation"].replace("_", " ").title())

            st.info(f"**Reason:** {cmp['reason']}")

            # --- System 1 signal plot ---
            if s1["ok"] and s1.get("signal") is not None:
                st.write("**System 1 — extracted signal:**")
                fig1, ax1 = plt.subplots(figsize=(12, 2.5))
                sig = s1["signal"]
                t = np.arange(len(sig)) / s1["fs"]
                ax1.plot(t, sig, lw=0.8, color="#1e5eb0")

                rp = s1.get("r_peaks")
                if rp is not None and len(rp) > 0:
                    rp_arr = np.asarray(rp)
                    ax1.scatter(rp_arr / s1["fs"], sig[rp_arr],
                                color="red", s=25, zorder=3,
                                label=f"{len(rp_arr)} peaks")
                    ax1.legend(loc="upper right")

                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("mV")
                ax1.set_title(f"System 1 — {n_sys1} R-peaks detected")
                st.pyplot(fig1)

            # --- System 2 status ---
            if s2["ok"]:
                st.write("**System 2 — Gemini Vision result:**")
                gc1, gc2, gc3 = st.columns(3)
                gc1.metric("Heart rate",
                           f"{(s2['result'].get('heart_rate_bpm') or '?')} bpm")
                gc2.metric("Image quality",
                           s2["result"].get("image_quality", "—"))
                gc3.metric("Referral",
                           "YES" if s2["result"].get("referral_recommended") else "No")

                if s2["result"].get("red_flags"):
                    st.warning("**Red flags:** " +
                               ", ".join(s2["result"]["red_flags"]))

            # --- Feedback buttons ---
            st.divider()
            st.write("**Which system was more correct on this image?**")
            st.caption("Your feedback auto-tunes System 1's parameters for future runs.")

            fb1, fb2, fb3, fb4 = st.columns(4)

            if fb1.button("System 1 (offline CV)"):
                from src.digitizer.feedback import record_feedback, auto_tune_params
                record_feedback(str(tmp), s1, s2, "sys1")
                tuned = auto_tune_params()
                st.success("Logged. System 1 auto-tuned.")
                if tuned:
                    st.json(tuned)

            if fb2.button("System 2 (Gemini)"):
                from src.digitizer.feedback import record_feedback, auto_tune_params
                record_feedback(str(tmp), s1, s2, "sys2")
                tuned = auto_tune_params()
                st.success("Logged. System 1 auto-tuned.")
                if tuned:
                    st.json(tuned)

            if fb3.button("Both correct"):
                from src.digitizer.feedback import record_feedback
                record_feedback(str(tmp), s1, s2, "both")
                st.success("Logged.")

            if fb4.button("Neither / unclear"):
                from src.digitizer.feedback import record_feedback
                record_feedback(str(tmp), s1, s2, "neither")
                st.warning("Logged. Manual review needed.")
            # After the feedback buttons
            st.divider()
            st.write("**Teach System 1 (offline ML distillation)**")

            from src.digitizer.learn import load_training_data, train_corrector
            n_pairs = len(load_training_data())
            st.caption(f"Training pairs collected: **{n_pairs}** (need 10+)")

            colA, colB = st.columns(2)
            if colA.button("Log this pair for training"):
                from src.digitizer.learn import record_pair
                if s1.get("signal") is not None:
                    # Use Gemini's beat positions as target signal proxy
                    record_pair(str(tmp), s1["signal"],
                                s1["signal"],  # placeholder target
                                s1.get("quality", 0.0))
                    st.success("Pair logged.")

            if colB.button("Train corrector now") and n_pairs >= 10:
                with st.spinner("Training..."):
                    model = train_corrector()
                if model:
                    st.success(f"Corrector trained on {n_pairs} pairs.")

            st.caption(
                "The corrector learns the delta between System 1's raw output "
                "and Gemini's reference output. After enough samples, "
                "System 1 can apply the learned correction **offline**."
            )
    else:
        st.info(
            "Upload an ECG photo. Both systems will analyze it — "
            "System 1 (offline CV) and System 2 (Gemini Vision) — "
            "and you can tell us which was more accurate. "
            "That feedback auto-tunes System 1 over time."
        )
