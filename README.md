# Turbofan PHM: sensor anomaly detection + RAG fault-diagnosis assistant

> **Status (November 2026)**
>
> **Part 1 – Anomaly detection: complete.** Models, evaluation protocol and final test results below.
>
> **Part 2 – RAG diagnosis assistant: in progress.** Design and roadmap in [Roadmap](#roadmap).

Early anomaly detection on NASA C-MAPSS turbofan run-to-failure data, comparing classical
process-monitoring methods with an LSTM autoencoder under an evaluation protocol fixed before
the models were compared, in which no test-set information enters any choice (split, scaling,
model settings, threshold or alarm rule). The second part, in progress, is designed to turn an
alarm into diagnostic support: the deviating sensors of the alarmed engine will be matched with
aviation maintenance documentation to suggest likely causes and checks, with citations.

**Main result (FD001, held-out test set):** on validation, the degradation signal was captured
mainly by the 2nd principal component of healthy sensor data. A Hotelling T² score on
2 components (a setting chosen on the validation engines, reported as *tuned*) alarmed on all
25 test engines that end near failure (RUL ≤ 30 at their last cycle); for half of them the first
alarm came at least 71 cycles (flights) before failure, with early alarms on 2 of 100 test
engines. Among models with at most 2% early alarms, this was the longest median lead time.
Compared with distance and Isolation Forest, its AUROC was higher in the 60–90 RUL band
(0.772 vs 0.689–0.726) and similar in the 30–60 band (0.927 vs 0.918–0.920). A rolling-mean
version gave similar values (0.928, 0.779, 74 cycles) with 5% early alarms. The T² setting fixed
in advance (95% variance, 12 of 14 components) scored below distance (test AUROC 30–60: 0.854 vs
0.918). Reconstruction-error scores (PCA SPE, window PCA SPE, LSTM autoencoder) scored lower.
No confidence intervals were computed; with 25 near-failure engines, differences of about
0.01 AUROC should not be read as real differences.

| Part | Status |
|---|---|
| Data pipeline, EDA, engine-level split | ✅ done |
| Evaluation protocol | ✅ done |
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
- **Sensors:** names follow Table 2 of the paper (T24, T30, T50, P30, Ps30, Nf, Nc, ...).

## Approach

### Exploratory analysis (`notebooks/`)

- **Sensor selection:** 6 sensors are constant in FD001 (T2, P2, epr, farB, Nf_dmd, PCNfR_dmd)
  and P15 takes only two values with no trend. The remaining **14 sensors** are used.
- **Healthy window:** the first **30 cycles** of each engine. This uses no failure information,
  so it is applicable in the field, and gives every engine equal weight. A RUL-based window
  (e.g. RUL > 150) was rejected: it needs future knowledge and gives 2 to 211 rows per engine.
  Over the first 30 cycles, the average drift per sensor (mean over engines) is at most
  0.11 healthy standard deviations.
- **Noise structure:** in healthy cycles, lag-1 autocorrelation is about 0 for every sensor
  (−0.06 to 0.00), consistent with white noise.
- **Between-engine variance:** 28–86% of healthy variance per sensor is between engines. The
  dataset description states that engines start with different, unknown initial wear and
  manufacturing variation.

### Evaluation protocol (`src/evaluate.py`)

- **Split by engine, never by row:** 80 fit engines / 20 validation engines
  (`results/splits/FD001.json`). Rows of one engine share its initial offset and degradation
  path; a row split would leak.
- **Training data:** healthy cycles of the 80 fit engines only. The scaler is fit on the same
  data.
- **Threshold:** 99th percentile of scores on the validation engines' healthy cycles. For the
  LSTM autoencoder the same cycles also drive early stopping, so its threshold is set on data the
  model was selected on. This can bias the threshold downward (more early alarms on new engines);
  the size of the effect was not measured.
- **Alarm:** score above threshold for k consecutive cycles. k is the smallest value in
  {1, 2, 3, 5} that gives no early alarm on validation, chosen per model (single-cycle models:
  k = 2–3; window-based models: mostly k = 5).
- **Metrics:**
  - *early alarm rate* (false alarms): share of engines that alarm within their first 30 cycles,
  - *detection rate*: share of engines ending near failure (RUL ≤ 30 at their last cycle) that
    alarm after cycle 30,
  - *lead time*: RUL at the first alarm; median over the detected engines only,
  - *banded AUROC*: healthy cycles (cycles 1–30; 10–30 for window-based models) vs cycles after
    cycle 30 with RUL in 0–30, 30–60 or 60–90. The 0–30 band is close to 1.0 for the stronger
    models, so the earlier bands separate the methods.
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
| Rolling | 10-cycle moving average per sensor | Distance or 2-component T² |
| LSTM autoencoder | last 10 cycles | mean reconstruction error of the window |

- PCA keeps the components that explain 95% of healthy variance (12 of 14 on FD001). A
  2-component T² was selected on validation (highest AUROC 30–60) after a 1–12 component sweep
  (`notebooks/03_base_checks.ipynb`), and is reported separately as *tuned*.
- LSTM autoencoder: encoder LSTM → 8-dimensional summary → decoder LSTM → linear layer.
  Trained on healthy windows; early stopping on validation-healthy reconstruction loss only
  (training stops after 10 epochs without improvement). The epoch cap was raised from 300 to
  1000 because the first runs reached the cap before early stopping triggered; the change used
  only validation-healthy loss. Three seeds, reported as mean ± std.
- Window length is 10 cycles so that 21 windows per engine fit inside the healthy window.
- All window features are causal: the score at cycle t uses cycles t−9 … t only.

## Results

FD001, metrics as defined under *Evaluation protocol*. Test detection is counted over the
25 test engines that end near failure, early alarms over all 100 test engines. On validation,
early alarms are 0% for every model, since k is chosen to give none there. When detection is
low, the median lead time rests on few engines and is not comparable with the other rows.
No value is highlighted (see [Limitations](#limitations)).

| Model | Val AUROC 30–60 | Test AUROC 30–60 | Test AUROC 60–90 | Test detection | Test median lead | Test early alarms |
|---|---|---|---|---|---|---|
| PCA T², 2 components *(tuned on val)* | 0.962 | 0.927 | 0.772 | 25/25 | 71 | 2% |
| Rolling PCA T², 2 components *(tuned on val)* | 0.963 | 0.928 | 0.779 | 24/25 | 74 | 5% |
| Isolation Forest | 0.937 | 0.920 | 0.726 | 24/25 | 47.5 | 0% |
| Distance | 0.928 | 0.918 | 0.689 | 24/25 | 47 | 0% |
| Rolling distance | 0.915 | 0.907 | 0.667 | 24/25 | 46 | 2% |
| PCA T², 95% variance | 0.891 | 0.854 | 0.674 | 21/25 | 50 | 0% |
| LSTM autoencoder (3 seeds) | 0.804 ± 0.006 | 0.759 ± 0.012 | 0.562 ± 0.007 | 24, 22, 21 /25 | 45 ¹ | 1.7% (4, 1, 0%) |
| Window PCA T² | 0.803 | 0.756 | 0.586 | 20/25 | 57.5 | 10% |
| Window PCA SPE | 0.564 | 0.588 | 0.519 | 2/25 | 26.5 ² | 0% |
| PCA SPE | 0.536 | 0.543 | 0.518 | 2/25 | 31 ² | 0% |

¹ Mean of the three per-seed medians (45, 44.5, 46).
² Median over the 2 detected engines only; not comparable with the other rows.

Full tables, including k, thresholds and lead-time quartiles:
`results/val_results_FD001.csv`, `results/test_results_FD001.csv`.

## Findings

Observed on FD001 (20 validation, 100 test engines). Measured results are kept separate from
their interpretation where one is given.

1. **With 12 of 14 components kept, T² detected degradation and SPE did not.**
   *Measured:* near failure (validation engines, last 30 cycles) the sensors shift by
   2.2–4.7 healthy standard deviations (e.g. T50 and Ps30 about 3.4). With the 95%-variance PCA
   (12 of 14 components), single-cycle and rolling T² separate degraded cycles well (test AUROC
   30–60: 0.854–0.928) while SPE is close to chance (0.543).
   *Interpretation:* keeping 12 of 14 components leaves a 2-dimensional residual, so SPE only
   sees shifts in the two lowest-variance healthy directions. The component sweep (point 2)
   suggests the signal lies mainly along the 2nd component instead.
2. **Much of the degradation signal seems to lie along one principal direction.**
   *Measured (validation sweep, 1–12 components):* T² is weakest with 1 component (AUROC 0.687),
   best with 2 (0.962) and declines with every component added after that (0.948 with 3,
   0.891 with 12). With 1 component, SPE does better than T² (0.847); once the 2nd component is
   inside the model subspace, SPE drops to 0.579.
3. **The LSTM autoencoder did not outperform the linear methods in this setting.**
   *Measured:* test AUROC 30–60 of 0.759 ± 0.012 over 3 seeds, close to window PCA T² (0.756),
   above SPE (0.543) and below the other single-cycle scores (0.854–0.927).
4. **The validation-tuned gain was smaller on test.**
   *Measured:* the 2-component T² led distance by 0.034 AUROC (30–60) on validation and by
   0.009 on test, while keeping a lead in the 60–90 band (0.772 vs 0.689) and in median lead time
   (71 vs 47 cycles).
   *Interpretation:* consistent with selection bias from choosing the setting on 20 validation
   engines; the remaining 30–60 difference (0.009) is small for 25 near-failure engines, and no
   confidence interval was computed.
5. **Window-based scores raised more early alarms on test.**
   *Measured:* engines alarming in their first 30 cycles, out of 100: window PCA T² 10,
   rolling T² 5, rolling distance 2, LSTM autoencoder 4 / 1 / 0 (three seeds), single-cycle
   models 0–2. These counts are small and their cause was not tested.

## Limitations

- Simulated data; one operating condition and one fault mode (FD001).
- Only 25 test engines end near failure and no confidence intervals were computed; AUROC
  differences around 0.01 should not be read as real differences.
- No ground-truth anomaly labels: RUL bands and alarm timing are proxies.
- Validation has 20 engines; thresholds and k are estimated from 20 × 30 healthy cycles
  (20 × 21 windows for window-based models).
- The 30-cycle healthy window is not guaranteed healthy: for 10 short-lived training engines it
  includes cycles with RUL below 125 (3.6% of its rows; lowest RUL 98).

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

- Hotelling T² on the autoencoder's latent representation.
- Per-engine baselines, motivated by the between-engine variance.
- Healthy-window sensitivity (20 and 50 cycles).
- FD003 (two fault modes, where diagnosis has a real choice to make) and FD004 (six operating
  conditions, condition-wise normalization).

## Setup and reproduction

Open in GitHub Codespaces (Code → Codespaces → Create). The dev container installs Python 3.12,
CPU-only PyTorch and the pinned dependencies; `python scripts/smoke_test.py` checks the
environment. `GROQ_API_KEY` (Codespaces secret) is needed only for the RAG part.

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
│   ├── features.py       # healthy window, scaler on healthy fit engines
│   ├── views.py          # cycle / flattened-window / rolling inputs (causal)
│   ├── baselines.py      # distance, PCA SPE and T², Isolation Forest
│   ├── lstm_ae.py        # LSTM autoencoder with early stopping
│   ├── evaluate.py       # threshold, k-consecutive alarms, metrics
│   └── run_baselines.py  # runs every model on val or test
├── notebooks/            # EDA, feature diagnostics, component sweep
├── scripts/smoke_test.py
├── tests/                # unit tests (alarm logic, windows, models, split, features)
├── results/              # split file and result tables
└── docs/maintenance/     # documents for the RAG index (empty until Part 2)
```

## References

- A. Saxena, K. Goebel, D. Simon, N. Eklund, *Damage Propagation Modeling for Aircraft Engine
  Run-to-Failure Simulation*, PHM08, 2008.
- Federal Aviation Administration, *Aviation Maintenance Technician Handbook – Powerplant*,
  FAA-H-8083-32B, 2023.