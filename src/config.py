from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = RESULTS / "models"
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

# Healthy window: first N cycles of each engine (deployment-realistic; no RUL used).
HEALTHY_CYCLES = 30
HEALTHY_CYCLES_SENSITIVITY = (20, 30, 50)

for _p in (DATA_RAW, DATA_PROCESSED, RESULTS, MODELS):
    _p.mkdir(parents=True, exist_ok=True)
