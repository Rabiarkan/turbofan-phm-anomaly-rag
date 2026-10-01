"""Build model inputs: selected sensors, healthy window, scaler fitted on healthy fit units."""

import json
from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (
    DATA_PROCESSED,
    FEATURE_SENSORS,
    HEALTHY_CYCLES,
    SENSOR_NAMES,
    SPLITS,
    SUBSET,
)


@dataclass
class Prepared:
    fit_healthy: pd.DataFrame  # fit units, first N cycles: train models here
    val_healthy: pd.DataFrame  # val units, first N cycles: set thresholds here
    val: pd.DataFrame  # val units, full life: tune alarm rule / compare configs
    test: pd.DataFrame  # test units: final evaluation, used once
    scaler: StandardScaler


def load_subset(subset: str = SUBSET) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train = pd.read_parquet(DATA_PROCESSED / f"train_{subset}.parquet")
    test = pd.read_parquet(DATA_PROCESSED / f"test_{subset}.parquet")
    split = json.loads((SPLITS / f"{subset}.json").read_text())
    return train.rename(columns=SENSOR_NAMES), test.rename(columns=SENSOR_NAMES), split


def healthy(df: pd.DataFrame, n_cycles: int = HEALTHY_CYCLES) -> pd.DataFrame:
    return df[df["cycle"] <= n_cycles]
    # The condition “cycle <= N” is based on the assumption that each motor's cycles start at 1 and
    # proceed sequentially without skipping any.This assumption was confirmed in the source document


def prepare(subset: str = SUBSET, n_cycles: int = HEALTHY_CYCLES) -> Prepared:
    train, test, split = load_subset(subset)
    fit = train[train["unit"].isin(split["fit"])]
    val = train[train["unit"].isin(split["val"])]

    fit_healthy = healthy(fit, n_cycles)
    scaler = StandardScaler().fit(fit_healthy[FEATURE_SENSORS])

    def scaled(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[FEATURE_SENSORS] = scaler.transform(df[FEATURE_SENSORS])
        return out

    return Prepared(
        fit_healthy=scaled(fit_healthy),
        val_healthy=scaled(healthy(val, n_cycles)),
        val=scaled(val),
        test=scaled(test),
        scaler=scaler,
    )
