\# Project Sentinel — The Pitch



\## The Problem

\- CVD kills 17.9M/year globally; \~30% of deaths in Pakistan

\- Pakistan has \~2,000 cardiologists for 240 million people (1:120,000)

\- Rural clinics have ECG machines but no specialists to read them

\- Result: heart attacks get missed until it's too late



\## Our Solution

An offline ECG triage system that:

1\. Loads a raw ECG (MIT-BIH format)

2\. Preprocesses + segments beats automatically

3\. Classifies each beat into 5 AAMI classes (N/S/V/F/Q)

4\. \*\*Refers to a cardiologist\*\* when uncertain (< 0.70 confidence)

&#x20;  or when the class is dangerous (V/S/F/Q)

5\. Runs on a laptop. No internet. No cloud. No privacy risk.



\## The Data (Person A)

\- \*\*MIT-BIH Arrhythmia Database\*\* — 48 records, \~110,000 annotated beats

\- Gold-standard, cardiologist-verified

\- We built a full industrial pipeline:

&#x20; - Config-driven, logged, tested (10/10 tests passing)

&#x20; - Patient-separated splits (no data leakage)

&#x20; - Class imbalance: 113:1 (N vs F)

&#x20; - Cache: `data/processed.npz`

&#x20; - Splits: 33 train / 7 val / 8 test records



\## The Model (Person B)

\- Feature-based Random Forest (fast, explainable)

\- `class\_weight='balanced'` for imbalance

\- Per-class recall reported (not just accuracy)

\- Isotonic calibration for meaningful confidence



\## The Demo (Person C)

\- Streamlit UI: upload ECG → live prediction

\- Red "REFER TO CARDIOLOGIST" banner on uncertain/dangerous beats

\- Before/after calibration diagrams

\- Confusion matrix from real test split



\## Impact

\- \*\*Cost:\*\* $0 (runs on any laptop)

\- \*\*Speed:\*\* < 5 ms per beat

\- \*\*Deployment:\*\* rural clinic with no internet

\- \*\*Safety:\*\* refuses to guess when uncertain



\## Why We Stand Out

\- Real dataset, real ML, real evaluation

\- Patient-separated splits (most teams miss this)

\- Confidence calibration + ECE (most teams don't know what this is)

\- Safety-first design (refer instead of diagnose)

