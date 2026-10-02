"""Shared evaluation protocol for every anomaly score.

Input: a DataFrame with columns unit, cycle, rul, score (higher = more anomalous).
The threshold comes from val healthy cycles only; the same rules apply to every model.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import (
    ALARM_K_CANDIDATES,
    ALARM_QUANTILE,
    AUROC_BANDS,
    HEALTHY_CYCLES,
    NEAR_FAILURE_RUL,
)


def threshold(healthy_scores: pd.Series, q: float = ALARM_QUANTILE) -> float:
    return float(np.quantile(healthy_scores, q))


def run_length(df: pd.DataFrame, thr: float) -> pd.Series:
    """Length of the current streak of score > thr at each cycle, per unit."""
    df = df.sort_values(["unit", "cycle"])
    exceed = df["score"] > thr
    streak_id = (~exceed).groupby(df["unit"]).cumsum()
    return exceed.astype(int).groupby([df["unit"], streak_id]).cumsum()


def first_alarm_cycle(df: pd.DataFrame, thr: float, k: int, after: int = 0) -> pd.Series:
    """First cycle > `after` where the streak reaches k; NaN if never. Indexed by unit."""
    df = df.sort_values(["unit", "cycle"])
    hit = (run_length(df, thr) >= k) & (df["cycle"] > after)
    cycles = df["cycle"].where(hit)
    return cycles.groupby(df["unit"]).min().reindex(df["unit"].unique())


def auroc(df: pd.DataFrame, lo: int, hi: int) -> float:
    """Threshold-free: healthy cycles vs cycles with lo < RUL <= hi (outside the healthy window)."""
    healthy = df["cycle"] <= HEALTHY_CYCLES
    band = (df["rul"] > lo) & (df["rul"] <= hi) & ~healthy
    if not band.any() or not healthy.any():
        return float("nan")  # undefined: one of the two groups is empty
    sel = healthy | band
    return float(roc_auc_score(band[sel], df.loc[sel, "score"]))


def evaluate(df: pd.DataFrame, thr: float, k: int) -> dict:
    df = df.sort_values(["unit", "cycle"])
    units = df["unit"].unique()

    early = first_alarm_cycle(df[df["cycle"] <= HEALTHY_CYCLES], thr, k)
    detect = first_alarm_cycle(df, thr, k, after=HEALTHY_CYCLES)

    end_rul = df.groupby("unit")["rul"].min()
    near_units = end_rul[end_rul <= NEAR_FAILURE_RUL].index
    rul = df.set_index(["unit", "cycle"])["rul"]
    det_near = detect.loc[near_units].dropna()
    lead = pd.Series([rul[(u, int(c))] for u, c in det_near.items()], dtype=float)

    return {
        "k": k,
        "threshold": round(thr, 4),
        "units": len(units),
        "early_alarm_rate": round(float(early.notna().mean()), 3),
        "near_failure_units": len(near_units),
        "detection_rate": round(len(det_near) / max(len(near_units), 1), 3),
        "lead_median": float(lead.median()) if len(lead) else np.nan,
        "lead_q25": float(lead.quantile(0.25)) if len(lead) else np.nan,
        "lead_q75": float(lead.quantile(0.75)) if len(lead) else np.nan,
    } | {f"auroc_{lo}_{hi}": round(auroc(df, lo, hi), 3) for lo, hi in AUROC_BANDS}


def select_k(val_df: pd.DataFrame, thr: float, candidates=ALARM_K_CANDIDATES) -> int:
    """Pre-registered rule: smallest k with no early alarm on val; else the largest k."""
    for k in candidates:
        if evaluate(val_df, thr, k)["early_alarm_rate"] == 0:
            return k
    return max(candidates)
