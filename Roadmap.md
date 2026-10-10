# Project Sentinel — Master Roadmap

**Team:** Thiraj.io
**Event:** AICON'26 Build With AI
**Domain:** Health Operations
**Slot:** 11th October 2026, 11:20 AM · SEECS Lecture Hall
**Total time:** ~22 hours remaining
**Progress:** ~5% (foundation only — the real work starts now)

---

## 🎯 What We're Building (Final Lock)

An **offline ECG arrhythmia triage system** for rural clinics in Pakistan.

Pipeline: Raw ECG → Preprocess → Segment beats → Extract features → Classify (5 AAMI
classes) → Calibrate confidence → Refer if uncertain/dangerous → Streamlit UI


**Killer feature:** The system says **"REFER TO CARDIOLOGIST"** when:
- Confidence < 0.70, OR
- Predicted class ∈ {V, S, F, Q} (all dangerous)

**Why it wins:** Real gold-standard data, patient-separated evaluation,
confidence calibration (ECE), offline-first, safety-first.

---

## 📊 Status — What's Done (Current)

### ✅ Person A — Data Platform (100% of Phase 1)

| File | Status |
|---|---|
| `src/constants.py` | ✅ AAMI mapping, paths |
| `src/config.py` + `configs/default.yaml` | ✅ config loader |
| `src/logger.py` | ✅ unified logging |
| `src/data_loader.py` | ✅ multi-lead + batch + benchmark + validation |
| `src/preprocess.py` | ✅ filter + segment + normalize |
| `src/splits.py` | ✅ patient-separated splits |
| `src/dataset.py` | ✅ cached dataset builder |
| `src/audit.py` | ✅ JSON quality report |
| `src/visualize.py` | ✅ 6 plot functions |
| `src/report.py` | ✅ JSON/MD/PDF export |
| `src/calibration.py` | ✅ isotonic + ECE + reliability diagrams |
| `src/demo.py` | ✅ demo beat sampler |
| `src/cli.py` | ✅ 6 subcommands |
| `src/smoke.py` | ✅ end-to-end proof |
| `tests/` | ✅ 10 passing tests |
| `data/processed.npz` | ✅ 109,464 beats cached |
| `data/splits/{train,val,test}.txt` | ✅ written |
| `docs/class_dist.png` | ✅ generated |
| `docs/dataset_report.json` | ✅ generated |

**Verified numbers:**
- 48 records, 109,464 beats, 5 classes
- N=90,604 (82.77%) · Q=8,042 (7.35%) · V=7,235 (6.61%)
- S=2,781 (2.54%) · F=802 (0.73%)
- Imbalance ratio: 112.97
- Splits: 33 train / 7 val / 8 test records
- Train=76,897 / Val=15,655 / Test=16,912 beats
- Full pipeline runs in 4.7s end-to-end

### ⏳ Person B — Features + Model (0%)

| Deliverable | Status |
|---|---|
| `src/features.py` | ❌ not started |
| `src/model.py` | ❌ not started |
| Trained `model.pkl` | ❌ not started |
| Metrics (`model_metrics.json`) | ❌ not started |

### ⏳ Person C — Training + UI (0%)

| Deliverable | Status |
|---|---|
| `src/train.py` | ❌ not started |
| `app.py` (Streamlit) | ❌ not started |
| Live demo with RED referral banner | ❌ not started |
| Confusion matrix + metrics panel | ❌ not started |

### ⏳ Integration + Demo (0%)

| Deliverable | Status |
|---|---|
| End-to-end smoke including model | ❌ not started |
| Calibration report with real model | ❌ not started |
| `run.py` one-command training | ❌ not started |
| `requirements.txt` | ❌ not started |
| `README.md` | ❌ not started |
| `.gitignore` | ❌ not started |
| `Makefile` | ❌ not started |
| Pitch deck content | ❌ not started |
| Demo rehearsal (×3) | ❌ not started |

---

## 🚀 Roadmap — What's Next (Future)

### Phase 2 — Core ML (Hours 0–6) — CRITICAL PATH

**Person B builds:**
- `src/features.py` — RR intervals, morphology, statistical features
- `src/model.py` — Random Forest classifier with confidence output
- `evaluate()` returns accuracy, per-class recall, confusion matrix, referral rate
- Trains `model.pkl` via `joblib`

**Person C builds:**
- `src/train.py` — uses A's `dataset.build_dataset()` + `splits.apply_split()`
- `app.py` — Streamlit UI with file upload OR record picker
- RED referral banner logic
- Matplotlib ECG waveform + prediction table

**Person A (you) does:**
- Prepare `run.py` connector (already drafted)
- Prepare `requirements.txt` (drafted)
- Prepare `README.md` (drafted)
- Write `.gitignore`, `Makefile`
- Write `docs/PITCH.md`

**Integration checkpoint:** when B and C push their branches, you merge
into `main` and run `python -m src.smoke` to verify nothing broke.

### Phase 3 — Calibration + Evaluation (Hours 6–10)

**Person A:**
- Wire `src/calibration.py` into the final evaluation flow
- Generate `docs/reliability_before.png` and `reliability_after.png`
- Compute ECE before/after isotonic calibration
- Save `docs/metrics.json` with all numbers

**Person B:**
- Report per-class recall (sklearn `recall_score`)
- Save confusion matrix to `docs/confusion.png`

**Person C:**
- Load `model_metrics.json` in Streamlit footer
- Display confusion matrix + recall table in UI

**Integration checkpoint:** every number in the pitch is traceable to a file.

### Phase 4 — Demo Drama (Hours 10–14)

**Person C + A:**
- Build a **live ECG streamer** — beats appear one by one, UI updates in real time
- Add `src/simulator.py` that feeds beats at 1 Hz from `sample_demo_beats()`
- Streamlit auto-refresh → waveform grows, predictions update live

**Person A:**
- Build `demo_run.py` — runs the full pipeline on stage with timing printouts
- Export a **one-page PDF** summary via `src/report.try_pdf()`

**Person B:**
- Save 3–5 "star beats" (one per class, hand-picked) for the demo
- Provide `predict_with_proba` for the UI

### Phase 5 — Polish (Hours 14–18)

- **UI polish:** Tailwind-equivalent Streamlit styling, colored badges, icons
- **Logging:** all runs write to `logs/sentinel.log`
- **Error handling:** missing dataset, empty upload, wrong file format
- **Docs:** auto-generated `PROJECT.md` with folder structure
- **Model card:** `docs/MODEL_CARD.md` — what the model does, limits, ethics

### Phase 6 — Rehearsal (Hours 18–22)

- 3× full rehearsal of 5-min pitch with live demo
- Time the demo: 3 min demo + 1.5 min problem/impact + 0.5 min Q&A prep
- Backup plan: if live demo breaks, switch to pre-recorded screenshots
- Print 3 copies of the pitch one-pager
- Charge all laptops, bring chargers, bring USB backups

---

## 🎤 Pitch Outline (5 min)

| Time | Section | Content |
|------|---------|---------|
| 0:00–0:30 | Hook | "30% of deaths in Pakistan are cardiac. 1 cardiologist per 120,000 people." |
| 0:30–1:00 | Problem | Rural clinics have ECG machines, no readers. |
| 1:00–2:00 | Solution | Live demo: upload ECG → predictions → red referral banner |
| 2:00–3:00 | Tech | AAMI 5-class RF classifier, patient-separated eval, ECE calibration |
| 3:00–4:00 | Numbers | Accuracy, per-class recall, referral rate, < 5 ms/beat |
| 4:00–4:30 | Impact | $0, offline, safe by design |
| 4:30–5:00 | Close | "If we don't build tools for the places that need them most, nobody will." |

---

## 📦 Deliverables Checklist (Final)

- [ ] `model.pkl` trained
- [ ] `model_metrics.json` with accuracy + per-class recall + confusion matrix
- [ ] `docs/reliability_before.png` + `reliability_after.png`
- [ ] `docs/confusion.png`
- [ ] `docs/feature_importance.png`
- [ ] `docs/class_dist.png` (already done)
- [ ] `docs/signal.png` (sample ECG waveform)
- [ ] Streamlit app works offline
- [ ] `python -m src.smoke` runs clean
- [ ] `run.py` trains model in one command
- [ ] `README.md` complete
- [ ] `requirements.txt` pinned
- [ ] `.gitignore` in place
- [ ] All 3 teammates pushed to their branches
- [ ] All branches merged to `main`
- [ ] Pitch rehearsed 3× with live demo
- [ ] Backup laptop + USB with model + screenshots

---

## 👥 Division of Work (Locked)

| Person | Owns | Do not touch |
|--------|------|--------------|
| **A** | `src/constants.py`, `src/config.py`, `src/logger.py`, `src/data_loader.py`, `src/preprocess.py`, `src/splits.py`, `src/dataset.py`, `src/audit.py`, `src/visualize.py`, `src/report.py`, `src/calibration.py`, `src/demo.py`, `src/cli.py`, `src/smoke.py`, `tests/`, `run.py`, `requirements.txt`, `README.md`, `.gitignore`, `Makefile`, `docs/PITCH.md` | B's features/model, C's train/app |
| **B** | `src/features.py`, `src/model.py`, `model.pkl`, `model_metrics.json` | A's pipeline files, C's train/app |
| **C** | `src/train.py`, `app.py`, `docs/confusion.png` | A's pipeline files, B's features/model |

**Rule:** if you need to change someone else's file, ask first. If A's
interface needs to change, A updates it and pings everyone.

---

## ⚠️ Risk Register

| Risk | Mitigation |
|------|-----------|
| B's features don't align with A's beat array | Contract is locked — `(n, 210)` in, `(n, features)` out |
| Model overfits to N class | `class_weight='balanced'`, report per-class recall |
| Streamlit demo crashes on stage | Pre-recorded video backup + pre-loaded sample |
| Git merge conflicts | One person per file, PR per feature, commit every 30 min |
| Python 3.15 wheel issues | Already resolved — all wheels installed |
| Time runs out | Phase 2 is critical path — B and C start NOW |

---

## 🕐 Live Timeline (from now)

| Time | Who | What |
|------|-----|------|
| Now +0h | B, C | Start writing their files |
| Now +0h | A | `.gitignore`, `Makefile`, `README.md`, `PITCH.md` |
| Now +2h | A | Check on B, C progress |
| Now +4h | All | Integration: A pulls B + C branches, runs smoke |
| Now +6h | All | First working end-to-end demo |
| Now +8h | A | Calibration report with real model |
| Now +10h | A, C | Live streamer + polish |
| Now +12h | All | SLEEP ROTATION (one at a time) |
| Now +16h | All | Final polish, bug fixes |
| Now +18h | All | Rehearse pitch 1× |
| Now +20h | All | Rehearse pitch 2×, fix timing |
| Now +21h | All | Rehearse pitch 3×, backup plans |
| Now +22h | All | Submit, sleep, be at venue early |

---

## 📢 Message to B and C (Copy-Paste)

> **Person A is 100% done with the data platform.** Load the dataset in one line:
> ```python
> import numpy as np
> d = np.load("data/processed.npz", allow_pickle=True)
> X, y = d["beats"], d["labels"]
> ```
> Splits (patient-separated, no leakage):
> ```python
> from src.splits import load_splits, apply_split
> data = apply_split(X, y, record_ids, load_splits())
> ```
> Class balance: **N=82.8%, Q=7.3%, V=6.6%, S=2.5%, F=0.7%** (113:1 imbalance).
> Use `class_weight='balanced'`. Report **per-class recall**, not just accuracy.
>
> Helpers already written for you:
> - `src.visualize` — 6 plot functions
> - `src.report.save_metrics_json` — save your metrics
> - `src.calibration` — isotonic calibration + ECE
> - `src.demo.sample_demo_beats` — for on-stage demo beats
>
> **Refer rule:** confidence < 0.70 OR class ∈ {V,S,F,Q} → show RED banner.
>
> Ping me when you have a working first version. I'll wire it into the smoke test.
>
> Read `ROADMAP.md` in the project root for the full plan.