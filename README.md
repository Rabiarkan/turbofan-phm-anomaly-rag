# Turbofan PHM: sensor anomaly detection + RAG fault-diagnosis assistant

> **Status (October 2026)**
>
> **Part 1 – Anomaly detection: complete.** Models, evaluation protocol and final test results below.
>
> **Part 2 – RAG diagnosis assistant: in progress.** Design and roadmap in [Roadmap](#roadmap).

Early anomaly detection on NASA C-MAPSS turbofan run-to-failure data, comparing classical
process-monitoring methods with an LSTM autoencoder under a leakage-free evaluation protocol fixed before the models were compared. The second part turns an alarm into diagnostic support: the deviating
sensors of the alarmed engine are matched with aviation maintenance documentation to suggest
likely causes and checks, with citations.

**Main result (FD001, held-out test set):** degradation appears to move largely *inside* the
healthy sensor correlation structure. Subspace distance (PCA Hotelling T²) detected it
best: for half of the engines that reach failure it warned more than 70 cycles (flights) ahead, with a 2% false-alarm rate
(2 of 100 test engines alarmed within their first 30 cycles). Reconstruction-error scores
(PCA SPE, window PCA, LSTM autoencoder) scored lower; possible reasons are discussed in
[Findings](#findings).

| Part | Status |
|---|---|
| Data pipeline, EDA, engine-level split | ✅ done |
| Evaluation protocol (fixed before comparison) | ✅ done |
| Classical baselines + LSTM autoencoder | ✅ done |
| Final test evaluation (FD001) | ✅ done, run once |
| RAG diagnosis assistant | 🚧 in progress |
| FD003 / FD004 | planned |

## Contents

- [Problem](#problem)
- [Data](#data)
- [Approach](#approach)
- [Results](#results)
- [Findings](#findings)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Setup and reproduction](#setup-and-reproduction)
- [Repository structure](#repository-structure)
- [References](#references)

## Problem

A fleet of engines degrades over time until failure. There are no "this cycle is anomalous"
labels, only the knowledge that each engine starts healthy. The task is therefore unsupervised:

1. learn what healthy sensor behaviour looks like,
2. score how far each new cycle is from it,
3. raise an alarm early enough to act, without false alarms on healthy engines.

This project covers detection (is something wrong?) and, in its second part, diagnostic
support (which component, what to check?). It does not estimate remaining useful life (RUL);
RUL is used only to evaluate how early alarms arrive.

## Data

NASA C-MAPSS turbofan run-to-failure simulation (Saxena, Goebel, Simon, Eklund, PHM08).

- **Source:** NASA Open Data Portal, `CMAPSSData.zip`
  (https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data),
  sha256 `74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f`, downloaded
  2026-09-29 and verified automatically by `python -m src.data`. No account needed.
  The portal lists no license; please cite the paper above.
- **Subset used:** FD001, one operating condition (sea level) and one fault mode (high-pressure
  compressor degradation). 100 training engines run to failure; 100 test engines stop before
  failure, with true RUL given.
- **Sensors:** names follow Table 2 of the paper (T24, T30, T50, P30, Ps30, Nf, Nc, ...). The
  column order is the commonly used assumption and is not stated explicitly in the paper or the
  dataset's readme; the observed relation between Ps30 and phi (= fuel flow / Ps30) is consistent
  with it.

## Approach

### Exploratory analysis (`notebooks/`)

- **Sensor selection:** 6 sensors are constant in FD001 (T2, P2, epr, farB, Nf_dmd, PCNfR_dmd;
  they reflect the single operating condition or controller set-points) and P15 takes only two
  values with no trend. The remaining **14 sensors** are used.
- **Healthy window:** the first **30 cycles** of each engine. This uses no failure information,
  so it is applicable in the field, and gives every engine equal weight. A RUL-based window
  (e.g. RUL > 150) was rejected: it needs future knowledge and gives 2 to 211 rows per engine.
  Within the first 30 cycles, sensor drift is at most 0.11 healthy standard deviations.
- **Noise structure:** lag-1 autocorrelation of healthy cycles is about 0 (consistent with white
  noise), so time information sits in the slow degradation trend, not in short-term dynamics.
- **Between-engine variance:** 28–86% of healthy variance per sensor is between engines
  (initial wear and manufacturing variation).

### Evaluation protocol (`src/evaluate.py`), fixed before the models were compared

- **Split by engine, never by row:** 80 fit engines / 20 validation engines
  (`results/splits/FD001.json`). Rows of one engine are near-duplicates; a row split would leak.
- **Training data:** healthy cycles of the 80 fit engines only. The scaler is fit on the same
  data.
- **Threshold:** 99th percentile of scores on the validation engines' healthy cycles.
- **Alarm:** score above threshold for k consecutive cycles. Rule: the smallest k in
  {1, 2, 3, 5} with no early alarm on validation, chosen per model (window-based scores are
  autocorrelated and need a larger k).
- **Metrics:**
  - *early alarm rate*: engines alarming within their first 30 cycles,
  - *detection rate*: engines ending near failure (RUL ≤ 30) that alarm after cycle 30,
  - *lead time*: RUL at the first alarm,
  - *banded AUROC*: healthy cycles vs cycles with RUL in 0–30, 30–60 and 60–90. The 0–30 band
    is easy (close to 1.0 for the stronger models), so the earlier bands separate the methods.
- **Test set:** evaluated once, after all models were final. Threshold and k come from
  validation; nothing is chosen on test.

### Models

| Name | Input | Score |
|---|---|---|
| Distance | one cycle (14 sensors) | mean squared z-score |
| PCA SPE | one cycle | squared residual outside the PCA subspace (Q statistic) |
| PCA Hotelling T² | one cycle | variance-scaled distance inside the PCA subspace |
| Isolation Forest | one cycle | negative `score_samples` |
| Window PCA (SPE, T²) | last 10 cycles, flattened (140 values) | as above |
| Rolling | 10-cycle moving average per sensor | Distance or T² |
| LSTM autoencoder | last 10 cycles | mean reconstruction error of the window |

- PCA keeps components for 95% of healthy variance (fixed in advance). A 2-component T² was
  selected on validation after a 1–12 component sweep and is reported separately as *tuned*.
- LSTM autoencoder: encoder LSTM → 8-dimensional summary → decoder LSTM → linear layer.
  Trained on healthy windows; early stopping on validation-healthy reconstruction loss only
  (patience 10). Three seeds, reported as mean ± std.
- Window length is 10 cycles so that 21 windows per engine fit inside the healthy window.
- All window features are causal: the score at cycle t uses cycles t−9 … t only.

## Results

FD001. AUROC compares healthy cycles with cycles in the given RUL band. **Lead time** is the
median RUL at the first alarm over the 25 test engines that end near failure (half of these engines
were warned at least this many cycles before failure). **Early alarms** (false alarms) are counted per engine: the
share of the 100 test engines that alarm within their first 30 cycles. On validation this is
optimistic by construction, since the threshold comes from the same cycles. Thresholds and k
come from the validation engines only.

| Model | Val AUROC 30–60 | Test AUROC 30–60 | Test AUROC 60–90 | Test median lead | Test early alarms |
|---|---|---|---|---|---|
| PCA T², 2 components *(tuned on val)* | 0.962 | **0.927** | **0.772** | **71** | 2% |
| Rolling PCA T², 2 components *(tuned on val)* | 0.963 | 0.928 | 0.779 | 74 | 5% |
| Isolation Forest | 0.937 | 0.920 | 0.726 | 47.5 | 0% |
| Distance | 0.928 | 0.918 | 0.689 | 47 | 0% |
| Rolling distance | 0.915 | 0.907 | 0.667 | 46 | 2% |
| PCA T², 95% variance | 0.891 | 0.854 | 0.674 | 50 | 0% |
| LSTM autoencoder (3 seeds, mean) | 0.804 ± 0.006 | 0.759 ± 0.012 | 0.562 | 45 | 1.7% |
| Window PCA T² | 0.803 | 0.756 | 0.586 | 57.5 | 10% |
| Window PCA SPE | 0.564 | 0.588 | 0.519 | 26.5 | 0% |
| PCA SPE | 0.536 | 0.543 | 0.518 | 31 | 0% |

Full tables, including k, thresholds and lead-time quartiles:
`results/val_results_FD001.csv`, `results/test_results_FD001.csv`.

## Findings

Observed on FD001 (20 validation, 100 test engines). Each point separates what was measured from
how it is interpreted.

1. **Degradation appears to stay largely inside the healthy correlation structure.**
   *Measured:* sensors that move together in healthy operation also shift together near failure
   (T50 and Ps30 by about 3.4 healthy standard deviations each); T² separates degraded cycles
   well (test AUROC 30–60: 0.85–0.93) while SPE is close to chance (0.54).
   *Interpretation:* the shift is mostly along healthy directions rather than a break in the
   correlation pattern.
2. **Much of the degradation signal seems to lie along one principal direction.**
   *Measured (validation sweep, 1–12 components):* T² is weakest with 1 component (AUROC 0.687),
   best with 2 (0.962) and declines steadily as components are added (0.891 with 12); with
   1 component, SPE does better than T² (0.847).
   *Interpretation (not tested directly):* the 2nd component carries much of the wear direction,
   while the 1st may reflect between-engine differences, which is consistent with 28–86% of
   healthy variance being between engines. Extra components likely add mostly noise.
3. **The LSTM autoencoder did not outperform the linear methods in this setting.**
   *Measured:* test AUROC 30–60 of 0.759 ± 0.012 over 3 seeds, close to window PCA T² (0.756),
   above SPE (0.543) and below single-cycle T² (0.854–0.927).
   *Possible explanation:* its score is a reconstruction error, an SPE-type statistic, which
   point 1 suggests is less suited to this degradation; and healthy cycles are close to white
   noise (lag-1 autocorrelation ≈ 0), so a time window may add little to learn. Its advantage
   over SPE may come from poor reconstruction outside the training range. These explanations
   were not tested separately.
4. **The validation-tuned gain was smaller on test.**
   *Measured:* the 2-component T² led distance by 0.034 AUROC (30–60) on validation and by
   0.009 on test, while keeping a lead in the 60–90 band (0.772 vs 0.689) and in median lead time
   (71 vs 47 cycles).
   *Interpretation:* consistent with selection bias from choosing the setting on 20 validation
   engines; with 25 near-failure test engines, the remaining 30–60 difference is not meaningful.
5. **Window-based scores tended to raise more early alarms on test.**
   *Measured:* window PCA T² 10%, rolling T² 5%, single-cycle models 0–2%.
   *Interpretation:* in line with overlapping windows producing autocorrelated scores, which the
   k-consecutive rule suppresses less effectively.

## Limitations

- Simulated data; one operating condition and one fault mode (FD001).
- Only 25 test engines end near failure; AUROC differences around 0.01 are not meaningful.
- No ground-truth anomaly labels: RUL bands and alarm timing are proxies.
- Validation has 20 engines; thresholds and k are estimated from 20 × 30 healthy cycles.

## Roadmap

### Part 2 – RAG diagnosis assistant (in progress)

The assistant takes the anomaly model's output as input, so detection and diagnosis form one
pipeline:

1. **Deviation report.** When an engine alarms, rank sensors by their scaled deviation at the
   alarm cycle and write a short report with physical names and directions
   (e.g. "T50 ↑ 3.4σ, Ps30 ↑ 3.4σ, P30 ↓ 2.9σ").
2. **Retrieval.** Index the FAA *Aviation Maintenance Technician Handbook – Powerplant*
   (FAA-H-8083-32B; turbine engine and engine maintenance chapters) and retrieve the passages
   closest to the report.
3. **Generation.** An LLM (Groq API) proposes likely causes and checks, citing handbook pages.
   Output is framed as decision support; the maintenance engineer decides.
4. **Evaluation.** Retrieval recall@k on a small hand-labelled query set, and a check that every
   generated claim cites a retrieved passage. C-MAPSS has no diagnosis labels, so diagnostic
   correctness itself cannot be scored; this limitation will be stated with the results.

Checklist:

- [ ] Deviation report from alarm cycles (`src/deviation.py`)
- [ ] Handbook ingestion and chunking with page metadata
- [ ] Embedding index and retrieval
- [ ] Generation with citations
- [ ] Retrieval evaluation and example outputs in this README

### Later

- T² in the autoencoder's latent space (the non-linear counterpart of the best linear score).
- Per-engine baselines, motivated by the between-engine variance.
- Healthy-window sensitivity (20 and 50 cycles).
- FD003 (two fault modes, where diagnosis has a real choice to make) and FD004 (six operating
  conditions, condition-wise normalization).

## Setup and reproduction

Open in GitHub Codespaces (Code → Codespaces → Create). The dev container installs Python 3.12,
CPU-only PyTorch and the pinned dependencies, then runs an environment smoke test.
`GROQ_API_KEY` (Codespaces secret) is needed only for the RAG part.

```bash
python scripts/smoke_test.py            # environment check
python -m src.data                      # download + verify C-MAPSS, write FD001 parquet
python -m src.split                     # engine-level fit/val split (results/splits/)
python -m src.run_baselines             # all models on validation
python -m src.run_baselines FD001 test  # final test run (done once)
python -m pytest -q                     # unit tests
```

## Repository structure

```
├── .devcontainer/        # Codespaces: Python 3.12, CPU torch, post-create checks
├── src/
│   ├── config.py         # paths, sensors, healthy window, protocol and model settings
│   ├── data.py           # download, parse, RUL labels
│   ├── split.py          # engine-level fit/val split
│   ├── features.py       # sensor selection, healthy window, scaler on healthy fit engines
│   ├── views.py          # cycle / flattened-window / rolling inputs (causal)
│   ├── baselines.py      # distance, PCA SPE and T², Isolation Forest
│   ├── lstm_ae.py        # LSTM autoencoder with early stopping
│   ├── evaluate.py       # threshold, k-consecutive alarms, metrics
│   └── run_baselines.py  # runs every model on val or test
├── notebooks/            # EDA, feature diagnostics, checks
├── scripts/smoke_test.py
├── tests/                # unit tests (alarm logic, windows, models, split)
├── results/              # split file and result tables
└── docs/maintenance/     # documents for the RAG index
```

## References

- A. Saxena, K. Goebel, D. Simon, N. Eklund, *Damage Propagation Modeling for Aircraft Engine
  Run-to-Failure Simulation*, PHM08, 2008.
- P. Baldi, K. Hornik, *Neural networks and principal component analysis: learning from
  examples without local minima*, Neural Networks, 1989.
- Federal Aviation Administration, *Aviation Maintenance Technician Handbook – Powerplant*,
  FAA-H-8083-32B, 2023.
