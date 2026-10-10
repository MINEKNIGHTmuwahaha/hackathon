# Project Sentinel — Master Roadmap v2

**Team:** Thiraj.io
**Event:** AICON'26 Build With AI
**Domain:** Health Operations
**Slot:** 11th October 2026, 11:20 AM · SEECS Lecture Hall
**Time remaining:** ~18 hours
**Progress:** ~45% (data + model + demo all working; expansion + polish next)

---

## 🎯 What We're Building

An **offline ECG arrhythmia triage system** for rural clinics in Pakistan.

Pipeline:
Raw ECG → Preprocess → Segment → Features → Classify (AAMI 5) →
Calibrate → Refer if uncertain/dangerous → Streamlit UI

**Killer feature:** "REFER TO CARDIOLOGIST" when confidence < threshold
OR class ∈ {S, V, F, Q}.

---

## 📊 Current Status (v1 baseline — WORKING)

### ✅ Person A — Data Platform + Model (done)

| Component | Status |
|---|---|
| `src/constants.py` | ✅ AAMI mapping, paths |
| `src/config.py` + `configs/default.yaml` | ✅ config loader |
| `src/logger.py` | ✅ unified logging |
| `src/data_loader.py` | ✅ multi-lead, batch, benchmark |
| `src/preprocess.py` | ✅ filter + segment + normalize |
| `src/splits.py` | ✅ patient-separated splits |
| `src/dataset.py` | ✅ cached dataset builder |
| `src/audit.py` | ✅ JSON quality report |
| `src/visualize.py` | ✅ 6 plot functions |
| `src/report.py` | ✅ JSON/MD/PDF export |
| `src/calibration.py` | ✅ isotonic + ECE |
| `src/demo.py` | ✅ demo beat sampler |
| `src/features.py` | ✅ 41 features (RR + morphology + wavelet + QRS) |
| `src/model.py` | ✅ ArrhythmiaClassifier + EnsembleClassifier |
| `src/train.py` | ✅ TwoStageClassifier with model selection |
| `src/cli.py` | ✅ 6 subcommands |
| `src/smoke.py` | ✅ end-to-end proof |
| `app.py` (Streamlit) | ✅ working |
| `tests/` | ✅ 10 passing |
| `model.pkl` | ✅ trained |
| `model_metrics.json` | ✅ saved |

### 📊 Current metrics (v1, MIT-BIH only)

**Stage 1 (N vs not-N):** balanced_acc = 0.9919
**Stage 2 (S/V/F/Q):** balanced_acc = 0.9778

**Test-set performance (patient-separated):**

| Class | Recall | Status |
|---|---|---|
| N | 76.9% | ⚠️ below target |
| S | 57.5% | ⚠️ below target |
| V | **91.5%** | 🔥 excellent |
| F | **0.0%** | ❌ dataset too small |
| Q | **98.9%** | 🔥 excellent |
| **Accuracy** | **79.8%** | ⚠️ okay |
| **Referral rate** | 22.5% | reasonable |

### ⚠️ Known problems (v2 will fix)

1. **F recall = 0%** — only 802 F beats total, ~2 in test set
2. **N recall 77%** — stage 1 too aggressive on "not-N"
3. **S recall 57%** — S and N are morphologically similar
4. **Only one dataset** — MIT-BIH, single hospital, 1975–79

### ⏳ Person B — Features + Model
- ✅ Actually done by you (A) since B is behind

### ⏳ Person C — Training + UI
- ✅ Actually done by you (A) since C is behind

---

## 🚀 v2 Roadmap — Multi-Dataset Expansion

### Phase 1 — Add INCART (highest-value addition)

**Why INCART:**
- 75 records, 30 min each, 12-lead
- **Beat-level AAMI annotations** (same standard as MIT-BIH)
- 20,005 V beats (3x MIT-BIH), 1,939 S beats, **219 F beats** (helps F!)
- Different patient population → better generalization
- Different recording conditions → robust model

**What changes:**
- New file: `src/loaders/incart.py` — dataset-specific loader
- New file: `src/loaders/__init__.py` — registry of datasets
- Modify: `src/dataset.py` to iterate over all registered datasets
- Modify: `src/train.py` to accept `dataset_names=["mitdb", "incart"]`
- New file: `data/incart/` — 75 records downloaded

**Estimated time:** 1–1.5 hours

**Expected impact:**
- F recall: 0% → 10-25%
- S recall: 57% → 65-70%
- N recall: 77% → 82-85%
- Overall accuracy: 80% → 85-88%

### Phase 2 — Add SVDB (supraventricular booster)

**Why SVDB:**
- 78 records, 30 min each
- Focused on supraventricular arrhythmias
- **Has many more S beats** than MIT-BIH
- Same hospital as MIT-BIH but different patients

**What changes:**
- New file: `src/loaders/svdb.py`
- Update `src/dataset.py` registry
- Download 78 records to `data/svdb/`

**Estimated time:** 45 min

**Expected impact:**
- S recall: 65-70% → 75-80%
- Robustness: better

### Phase 3 — Cross-dataset evaluation

**New file:** `src/cross_eval.py`

Reports:
- Per-dataset test metrics
- Combined test metrics
- Confusion matrix per dataset
- Cross-dataset generalization (train on A, test on B)

**Why this impresses judges:**
- Proves the model generalizes across populations
- Shows understanding of domain shift
- Published papers love this

**Estimated time:** 45 min

### Phase 4 — Handle F class properly

**Options (in order):**
1. **Add more data first** (INCART 219 + ESTTDB 354) → if F recall > 15%, keep F
2. **If F still 0%** → merge F into V (clinical convention)
3. **Report honestly** in pitch either way

**Decision point:** after Phase 1 retrain

### Phase 5 — Calibration + confidence

**Add:**
- Isotonic calibration on stage 1 and stage 2 separately
- Report ECE before/after on the test set
- Adjust threshold based on calibration results

**Estimated time:** 30 min

### Phase 6 — Final training + demo

**Retrain** with all datasets + all improvements
**Update** `app.py` to show multi-dataset results
**Test** with `demo_run` and Streamlit
**Commit** the final model

**Estimated time:** 30 min

---

## 📦 New Files Required
sentinel-ecg/
├── src/
│ ├── loaders/
│ │ ├── init.py # dataset registry
│ │ ├── mitdb.py # wraps existing data_loader
│ │ ├── incart.py # NEW
│ │ └── svdb.py # NEW
│ ├── cross_eval.py # NEW — per-dataset evaluation
│ └── (existing files)
├── data/
│ ├── mitdb/ # ✅ already there
│ ├── incart/ # NEW — 75 records
│ └── svdb/ # NEW — 78 records
└── docs/
├── DATASETS.md # NEW — inventory of datasets
├── CROSS_VAL_RESULTS.md # NEW — table of per-dataset metrics
└── (existing docs)

---

## 📥 Datasets to Download

### INCART (Priority 1)
- **URL:** https://physionet.org/content/incartdb/1.0.0/
- **Size:** ~150 MB
- **Records:** 75, all 12-lead, 30 min
- **Sampling:** 257 Hz (need resample to 360 Hz)
- **Format:** wfdb (same as MIT-BIH)
- **Annotation:** AAMI compatible

**Download command:**
```bash
mkdir -p data/incart
cd data/incart
wget -r -np -nH --cut-dirs=3 -R "index.html*" https://physionet.org/files/incartdb/1.0.0/
cd ../..

---
