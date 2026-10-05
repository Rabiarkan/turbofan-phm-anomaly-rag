from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = RESULTS / "models"
SPLITS = RESULTS / "splits"
DOCS = ROOT / "docs" / "maintenance"

SEED = 42

# C-MAPSS
# Official NASA Open Data Portal copy (same files as the Kaggle mirror).
CMAPSS_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
CMAPSS_SHA256: str | None = "74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f"
# sha256 of the file downloaded on 2026-09-29.
SUBSET = "FD001"  # one operating condition, one fault mode (HPC degradation)

INDEX_COLS = ["unit", "cycle"]
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
COLUMNS = INDEX_COLS + SETTING_COLS + SENSOR_COLS  # 26 columns

# Sensor names from Saxena et al. (PHM08), Table 2. Column order is the commonly
# used assumption; neither the paper nor the dataset readme states it explicitly.
SENSOR_NAMES = dict(
    zip(
        SENSOR_COLS,
        "T2 T24 T30 T50 P2 P15 P30 Nf Nc epr Ps30 phi NRf NRc BPR farB htBleed "
        "Nf_dmd PCNfR_dmd W31 W32".split(),
        strict=True,
    )
)

# Chosen in notebooks/01_eda_fd001.ipynb: dropped 6 constant sensors
# (T2, P2, epr, farB, Nf_dmd, PCNfR_dmd) and P15 (two values, no trend).
FEATURE_SENSORS = "T24 T30 T50 P30 Nf Nc Ps30 phi NRf NRc BPR htBleed W31 W32".split()

# Healthy window: first N cycles of each engine (deployment-realistic; no RUL used).
HEALTHY_CYCLES = 30
HEALTHY_CYCLES_SENSITIVITY = (20, 30, 50)

for _p in (DATA_RAW, DATA_PROCESSED, RESULTS, MODELS, SPLITS):
    _p.mkdir(parents=True, exist_ok=True)

# Train/validation split of training engines (by unit, never by row).
VAL_FRACTION = 0.2


# Evaluation protocol (fixed before any model is scored)
ALARM_QUANTILE = 0.99  # threshold = this quantile of scores on val healthy cycles
ALARM_K_CANDIDATES = (1, 2, 3, 5)  # consecutive exceedances needed; chosen on val
NEAR_FAILURE_RUL = 30  # engines ending at RUL <= this must be detected
# AUROC of healthy cycles vs cycles in each RUL band: how early does the score separate?
AUROC_BANDS = ((0, 30), (30, 60), (60, 90))


# Baselines (pre-registered settings)
PCA_VARIANCE = 0.95  # components kept = smallest number explaining this share on fit healthy
IFOREST_TREES = 200
# Chosen on val after the pre-registered run (sweep 1..12, best auroc_30_60 for T2)
PCA_TUNED_COMPONENTS = 2

# Window length for window-based models (window PCA, rolling mean, LSTM-AE).
# Must fit inside the healthy window: 30 healthy cycles -> 21 healthy windows per engine.
WINDOW = 10


# LSTM autoencoder (pre-registered).
LSTM_HIDDEN = 8  # bottleneck size, smaller than the 14 sensors
LSTM_LR = 1e-3
LSTM_BATCH = 64
LSTM_MAX_EPOCHS = 1000  # raised from 300: cap was hit before early stopping (val-healthy loss only)
LSTM_PATIENCE = 10  # early stopping on val-healthy reconstruction loss
LSTM_SEEDS = (0, 1, 2)
