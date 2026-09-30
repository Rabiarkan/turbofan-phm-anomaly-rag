"""Download C-MAPSS (NASA Open Data Portal) and write one subset as parquet.

python -m src.data            # FD001
python -m src.data FD003
"""

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src.config import CMAPSS_SHA256, CMAPSS_URL, COLUMNS, DATA_PROCESSED, DATA_RAW, SUBSET


def download(url: str = CMAPSS_URL, sha256: str | None = CMAPSS_SHA256) -> Path:
    if list(DATA_RAW.rglob("train_FD001.txt")):
        return DATA_RAW
    zip_path = DATA_RAW / "CMAPSSData.zip"
    urllib.request.urlretrieve(url, zip_path)
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    print(f"sha256 {digest}  {zip_path.name}")
    if sha256 and digest != sha256:
        zip_path.unlink()
        raise ValueError(f"checksum mismatch: expected {sha256}, got {digest}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_RAW / "CMaps")
    return DATA_RAW


def _find(name: str) -> Path:
    hits = list(DATA_RAW.rglob(name))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one {name} under {DATA_RAW}, found {len(hits)}")
    return hits[0]


def _read(path: Path) -> pd.DataFrame:
    # Space-separated with trailing spaces, no header.
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"{path.name}: {df.shape[1]} columns, expected {len(COLUMNS)}")
    df.columns = COLUMNS
    return df


def load(subset: str = SUBSET) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = _read(_find(f"train_{subset}.txt"))
    test = _read(_find(f"test_{subset}.txt"))
    rul_end = pd.read_csv(_find(f"RUL_{subset}.txt"), header=None).iloc[:, 0]

    # Train runs to failure: RUL = last cycle - current cycle.
    train["rul"] = train.groupby("unit")["cycle"].transform("max") - train["cycle"]

    # Test stops before failure; RUL_*.txt gives the true RUL at each unit's last cycle.
    n_test = test["unit"].nunique()
    if len(rul_end) != n_test:
        raise ValueError(f"RUL_{subset}.txt has {len(rul_end)} rows but test has {n_test} units")
    rul_end.index = range(1, len(rul_end) + 1)
    last = test.groupby("unit")["cycle"].transform("max")
    test["rul"] = test["unit"].map(rul_end) + (last - test["cycle"])
    return train, test


def main(subset: str = SUBSET) -> None:
    download()
    train, test = load(subset)
    train.to_parquet(DATA_PROCESSED / f"train_{subset}.parquet", index=False)
    test.to_parquet(DATA_PROCESSED / f"test_{subset}.parquet", index=False)
    print(
        f"{subset}: train {train.shape} ({train.unit.nunique()} units), "
        f"test {test.shape} ({test.unit.nunique()} units)"
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else SUBSET)
