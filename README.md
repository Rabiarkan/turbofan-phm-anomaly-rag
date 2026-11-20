# Turbofan PHM: sensor anomaly detection + RAG fault-diagnosis assistant

> **Status (November 2026)**
>
> **Part 1 – Anomaly detection: complete.** Models, evaluation protocol and test results below.
>
> **Part 2 – RAG diagnosis assistant: in progress.** The detector is saved; deviation report,
> retrieval and generation are next. See [Roadmap](#roadmap).

Early anomaly detection on NASA C-MAPSS turbofan run-to-failure data, comparing classical
process-monitoring methods with an LSTM autoencoder under one evaluation protocol for all models.
No test-set information enters any choice (split, scaling, model settings, threshold or alarm
rule). Part 2 will match the sensors that drove an alarm with aviation maintenance documentation
to suggest possible causes and checks, with citations.

**Main result (FD001, test set).** A Hotelling T² score on 2 principal components (chosen on
validation, reported as *tuned*) alarmed on 25 of the 25 test engines that end near failure
(RUL ≤ 30 at their last cycle), with a median lead time of 71 cycles. It alarmed early (within the
first 30 cycles) on 2 of 100 engines and in mid-life (RUL > 125) on 2 of 88 engines, with 1.39
alarm events per 1000 mid-life cycles. For comparison, distance and Isolation Forest had median
lead times of 47 and 47.5 cycles, no early alarms and 0.60 and 0.79 mid-life alarm events per
1000 cycles. Test AUROC 30–60 was 0.927 for the 2-component T², 0.918 for distance and 0.920 for
Isolation Forest; AUROC 60–90 was 0.772, 0.689 and 0.726. No confidence intervals were computed.

| Part | Status |
|---|---|
| Data pipeline, EDA, engine-level split | ✅ done |
| Evaluation protocol | ✅ done |
| Classical baselines + LSTM autoencoder | ✅ done |
| Test evaluation (FD001) | ✅ done |
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
3. raise an alarm before failure, without alarms on healthy engines.

This project covers detection and, in Part 2, diagnostic support. It does not estimate remaining
useful life (RUL); RUL is used only to evaluate when alarms arrive.

## Data

NASA C-MAPSS turbofan run-to-failure simulation (Saxena, Goebel, Simon, Eklund, PHM08).

- **Source:** NASA Open Data Portal, `CMAPSSData.zip`
  (https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data),
  sha256 `74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f`, downloaded
  2026-09-29; `python -m src.data` verifies it on download. No account needed.
  The portal lists no license; please cite the paper above.
- **Subset used:** FD001, one operating condition (sea level) and one fault mode (high-pressure
  compressor, HPC, degradation). 100 training engines run to failure; 100 test engines stop
  before failure, with true RUL given.
- **Sensors:** names and descriptions from Table 2 of the paper (T24, T30, T50, P30, Ps30, Nf,
  Nc, ...). The column order is the commonly used mapping; neither the paper nor the dataset
  readme states it explicitly.

## Approach

### Exploratory analysis (`notebooks/`)

- **Sensor selection:** 6 sensors are constant in FD001 (T2, P2, epr, farB, Nf_dmd, PCNfR_dmd)
  and P15 takes only two values with no trend. The remaining **14 sensors** are used.
- **Healthy window:** the first **30 cycles** of each engine. It uses no failure information and
  gives every engine the same number of rows. A RUL-based window (e.g. RUL > 150) was not used:
  it needs the failure time, leaves 7 of 100 engines with no rows and gives 2 to 211 rows to the
  others. Over the first 30 cycles, the average drift per sensor (mean over engines) is at most
  0.11 healthy standard deviations.
- **Noise structure:** in healthy cycles, lag-1 autocorrelation is between −0.06 and 0.00 for
  every sensor.
- **Between-engine variance:** 28–86% of healthy variance per sensor is between engines. The
  dataset description states that engines start with different, unknown initial wear and
  manufacturing variation.

### Evaluation protocol (`src/evaluate.py`)

- **Split by engine, never by row:** 80 fit engines / 20 validation engines
  (`results/splits/FD001.json`). Rows of one engine share its initial offset and degradation
  path; a row split would leak.
- **Training data:** healthy cycles of the 80 fit engines only; the scaler is fit on the same
  data.
- **Threshold:** 99th percentile of scores on the validation engines' healthy cycles. For the
  LSTM autoencoder the same cycles also drive early stopping, which can bias its threshold
  downward; the size of the effect was not measured.
- **Alarm:** score above threshold for k consecutive cycles. k is the smallest value in
  {1, 2, 3, 5} that gives no early alarm on validation, chosen per model.
- **Metrics.** After the 30-cycle healthy window, an engine's life is split at RUL = 125, used as
  a proxy for degradation onset (not a label):
  - *early alarm rate*: share of engines that alarm within their first 30 cycles,
  - *detection rate*: share of engines ending near failure (RUL ≤ 30 at their last cycle) with
    the alarm on at some cycle after cycle 30 with RUL ≤ 125,
  - *lead time*: RUL at that first cycle (at most 125); median over the detected engines,
  - *mid-life false alarms* (cycles after cycle 30 with RUL > 125): share of engines with at
    least one alarm event, and alarm events per 1000 cycles; an alarm event is the cycle at which
    a streak reaches k,
  - *banded AUROC*: healthy cycles (1–30; 10–30 for window-based models) vs cycles after
    cycle 30 with RUL in 0–30, 30–60 or 60–90. On test, the 0–30 band is 0.997–1.0 for five of
    the ten models, so the tables report the 30–60 and 60–90 bands.

  The earlier detection and lead definitions (any alarm after cycle 30) are kept as `legacy_*`
  columns in the result files.
- **Test set:** scored when all models were final (tag `part1-fd001`), and re-scored once to add
  the RUL-125 metrics, which were committed before the re-scoring. No threshold, k or model
  setting changed, and every earlier column was reproduced exactly.

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
  (`notebooks/03_base_checks.ipynb`) and is reported separately as *tuned*.
- LSTM autoencoder: encoder LSTM → 8-dimensional summary → decoder LSTM → linear layer, trained
  on healthy windows with early stopping on validation-healthy reconstruction loss (patience 10).
  The epoch cap was raised from 300 to 1000 because the first runs reached it before early
  stopping; the change used only validation-healthy loss. Three seeds, reported as mean ± std.
- Window length is 10 cycles, so 21 windows per engine fit inside the healthy window. All window
  features are causal: the score at cycle t uses cycles t−9 … t only.

## Results

FD001. Test detection is counted over the 25 test engines that end near failure, early alarms
over all 100 test engines, mid-life false alarms over the 88 test engines with cycles after
cycle 30 and RUL > 125. On validation, early alarms are 0% for every model, since k is chosen to
give none there.

| Model | Val AUROC 30–60 | Test AUROC 30–60 | Test AUROC 60–90 | Test detection | Test median lead | Test early alarms | Test mid-life false alarms (engines / per 1000 cycles) |
|---|---|---|---|---|---|---|---|
| PCA T², 2 components *(tuned on val)* | 0.962 | 0.927 | 0.772 | 25/25 | 71 | 2% | 2.3% / 1.39 |
| Rolling PCA T², 2 components *(tuned on val)* | 0.963 | 0.928 | 0.779 | 24/25 | 71 | 5% | 8.0% / 1.59 |
| Isolation Forest | 0.937 | 0.920 | 0.726 | 24/25 | 47.5 | 0% | 2.3% / 0.79 |
| Distance | 0.928 | 0.918 | 0.689 | 24/25 | 47 | 0% | 2.3% / 0.60 |
| Rolling distance | 0.915 | 0.907 | 0.667 | 24/25 | 46 | 2% | 3.4% / 0.79 |
| PCA T², 95% variance | 0.891 | 0.854 | 0.674 | 21/25 | 50 | 0% | 2.3% / 0.40 |
| LSTM autoencoder (3 seeds) | 0.804 ± 0.006 | 0.759 ± 0.012 | 0.562 ± 0.007 | 24, 21, 20 /25 | 36.8 ¹ | 1.7% (4, 1, 0%) | 9.5% / 1.79 ¹ |
| Window PCA T² | 0.803 | 0.756 | 0.586 | 19/25 | 45 | 10% | 15.9% / 2.97 |
| Window PCA SPE | 0.564 | 0.588 | 0.519 | 2/25 | 26.5 ² | 0% | 2.3% / 0.60 |
| PCA SPE | 0.536 | 0.543 | 0.518 | 2/25 | 31 ² | 0% | 0% / 0 |

¹ Mean over the three seeds: lead medians 41.5, 34, 35; mid-life false alarms 13.6%, 8.0%,
6.8% of engines and 2.78, 1.39, 1.19 events per 1000 cycles.
² Median over the 2 detected engines only; not comparable with the other rows.

Full tables, including k, thresholds, lead-time quartiles and the `legacy_*` columns:
`results/val_results_FD001.csv`, `results/test_results_FD001.csv`.

## Findings

Observed on FD001 (20 validation, 100 test engines). Measured results are kept separate from
their interpretation.

1. **With 12 of 14 components, T² separated degraded cycles better than SPE.**
   *Measured:* near failure (validation engines, last 30 cycles) the sensors shift by
   2.2–4.7 healthy standard deviations (e.g. T50 and Ps30 about 3.4). With the 95%-variance PCA,
   test AUROC 30–60 was 0.854 (single-cycle) and 0.756 (window) for T², and 0.543 and 0.588
   for SPE.
   *Interpretation:* keeping 12 of 14 components leaves a 2-dimensional residual, so SPE only
   sees shifts in the two lowest-variance healthy directions.
2. **In the validation sweep, T² scored best with 2 components.**
   *Measured (1–12 components, AUROC 30–60):* T² 0.687 with 1 component, 0.962 with 2, then lower
   with each added component (0.948 with 3, 0.891 with 12). SPE 0.847 with 1 component and 0.579
   with 2.
   *Interpretation:* the shift near failure may lie mainly along the 2nd principal component of
   healthy data; this was not tested directly.
3. **The LSTM autoencoder scored below most single-cycle scores.**
   *Measured:* test AUROC 30–60 of 0.759 ± 0.012 over 3 seeds, close to window PCA T² (0.756),
   above SPE (0.543) and below distance, Isolation Forest and both single-cycle T² scores
   (0.854–0.927).
   *Interpretation:* consistent with point 1 if the shift lies mostly inside the healthy
   subspace, which a reconstruction-error score sees little of.
4. **The validation-tuned gain over distance was smaller on test.**
   *Measured:* AUROC 30–60 difference between the 2-component T² and distance was 0.034 on
   validation and 0.009 on test; AUROC 60–90 was 0.772 vs 0.689 and median lead time 71 vs 47
   cycles on test. In mid-life both alarmed on 2 of 88 test engines, with 1.39 and 0.60 alarm
   events per 1000 cycles.
   *Interpretation:* the smaller test difference is consistent with selection bias from choosing
   the setting on 20 validation engines.
5. **Window PCA T², rolling T² and the LSTM autoencoder had the most false alarms on test.**
   *Measured:* early alarms (out of 100 engines): window PCA T² 10, rolling T² 5, rolling
   distance 2, LSTM autoencoder 4 / 1 / 0, single-cycle models 0–2. Engines with a mid-life false
   alarm: window PCA T² 15.9%, LSTM autoencoder 13.6 / 8.0 / 6.8%, rolling T² 8.0%, rolling
   distance 3.4%, window PCA SPE and single-cycle models 0–2.3%. The counts are small and their
   cause was not tested.
6. **Counting only alarms at RUL ≤ 125 shortened three median lead times.**
   *Measured:* median lead, any alarm after cycle 30 → only alarms at RUL ≤ 125: window PCA T²
   57.5 → 45 (detection 20 → 19 of 25), LSTM autoencoder 45.2 → 36.8 (mean over seeds), rolling
   T² 74 → 71. The other models were unchanged.
   *Interpretation:* for these three models, part of the earlier lead came from alarms at
   RUL > 125; they also had the highest mid-life false-alarm rates (point 5).

## Limitations

- Simulated data; one operating condition and one fault mode (FD001).
- RUL = 125 is a proxy for degradation onset, which varies by engine: an alarm counted as a
  mid-life false alarm may come from an engine that started degrading earlier.
- No confidence intervals: only 25 test engines end near failure and mid-life false-alarm rates
  rest on a few engines, so differences of about 0.01 AUROC or one or two engines should not be
  read as real differences.
- No ground-truth anomaly labels: RUL bands and alarm timing are proxies.
- Thresholds and k are estimated from 20 validation engines × 30 healthy cycles
  (× 21 windows for window-based models).
- The 30-cycle healthy window is not guaranteed healthy: for 10 short-lived training engines it
  includes cycles with RUL below 125 (3.6% of its rows; lowest RUL 98).

## Roadmap

### Part 2 – RAG diagnosis assistant (in progress)

The assistant takes the detector's output as input. Detector: the 2-component T², saved with its
scaler, threshold and k (`src/detector.py`; rebuilding it must reproduce threshold 5.7303 and
k = 3).

1. **Deviation report.** For an alarmed engine, average the last 10 cycles up to the alarm, rank
   sensors by their contribution to T² (contributions sum to the T² value), and report each
   sensor's shift from the engine's own first 30 cycles, e.g. "T50 (LPT outlet temperature)
   ↑ 3.4σ". Sensor descriptions from Table 2 are used so that queries contain the words of
   maintenance documents rather than C-MAPSS mnemonics. Nc and NRc move in opposite directions
   across engines near failure (`notebooks/02_features_analysis.ipynb`), which needs to be
   checked in the reports.
2. **Retrieval.** Index the FAA *Aviation Maintenance Technician Handbook – Powerplant*
   (FAA-H-8083-32B; turbine engine and maintenance chapters), chunked with page and section
   metadata. Hybrid retrieval: BM25 plus a small CPU embedding model, combined by reciprocal rank
   fusion. The PDF is downloaded and checksum-verified by a script, not committed.
3. **Generation.** An LLM (Groq API) returns structured output in which each proposed cause or
   check cites a retrieved chunk id. Citations not in the retrieved set are dropped; when
   retrieval finds no supporting passage, the output states that the evidence is insufficient.
   The output is decision support for a maintenance engineer.
4. **Evaluation.**
   - Retrieval: recall@k and MRR on a hand-labelled query set.
   - Faithfulness: whether each claim is supported by the passage it cites (LLM judge, checked
     against a manually reviewed sample).
   - Retrieval vs no retrieval: the same LLM without retrieved context.
   - Fault mode: FD001 has one known fault mode (HPC degradation), so the top-ranked cause can be
     checked against it. Engine-level fault labels are not provided, so finer diagnostic
     correctness cannot be scored.
   - Alarms in mid-life: the detector had 1.33 (validation) and 1.39 (test) alarm events per 1000
     mid-life cycles. Reports from these alarms are used to check the "evidence is insufficient"
     output.

   In FD001 every engine has the same fault mode, so deviation reports are expected to be
   similar across engines; comparing causes would need a subset with more than one fault mode,
   such as FD003.

## Setup and reproduction

Open in GitHub Codespaces (Code → Codespaces → Create). The dev container installs Python 3.12,
CPU-only PyTorch and the pinned dependencies; `python scripts/smoke_test.py` checks the
environment. `GROQ_API_KEY` (Codespaces secret) is needed only for Part 2.

```bash
python scripts/smoke_test.py            # environment check
python -m src.data                      # download + verify C-MAPSS, write FD001 parquet
python -m src.split                     # engine-level fit/val split (results/splits/)
python -m src.run_baselines             # all models on validation
python -m src.run_baselines FD001 test  # test set, with the choices frozen on validation
python -m src.detector                  # save the Part 2 detector (results/models/)
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
│   ├── evaluate.py       # threshold, k-consecutive alarms, detection, false alarms, AUROC
│   ├── run_baselines.py  # runs every model on val or test
│   └── detector.py       # saved 2-component T² detector for Part 2
├── notebooks/            # EDA, feature diagnostics, component sweep
├── scripts/smoke_test.py
├── tests/                # unit tests (alarm logic and metrics, windows, models, split, features, detector)
├── results/              # split file and result tables
└── docs/maintenance/     # documents for the RAG index (empty until Part 2)
```

## References

- A. Saxena, K. Goebel, D. Simon, N. Eklund, *Damage Propagation Modeling for Aircraft Engine
  Run-to-Failure Simulation*, PHM08, 2008.
- Federal Aviation Administration, *Aviation Maintenance Technician Handbook – Powerplant*,
  FAA-H-8083-32B, 2023.