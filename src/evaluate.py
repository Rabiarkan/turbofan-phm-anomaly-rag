"""Shared evaluation protocol for every anomaly score.

Input: a DataFrame with columns unit, cycle, rul, score (higher = more anomalous).
The threshold comes from val healthy cycles only; the same rules apply to every model.

Regions of an engine's life (R = onset: a proxy for degradation onset, not a label):
  healthy window  cycle <= HEALTHY_CYCLES            -> early alarms
  mid-life        cycle >  HEALTHY_CYCLES, RUL >  R  -> false alarms
  degradation     cycle >  HEALTHY_CYCLES, RUL <= R  -> detection and lead time

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
    ONSET_RUL,
    ONSET_RUL_SENSITIVITY,
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


def evaluate(df: pd.DataFrame, thr: float, k: int, onset: int = ONSET_RUL) -> dict:
    df = df.sort_values(["unit", "cycle"])
    early = first_alarm_cycle(df[df["cycle"] <= HEALTHY_CYCLES], thr, k)
    lead = lead_times(df, thr, k, onset)
    return (
        {
            "k": k,
            "threshold": round(thr, 4),
            "units": df["unit"].nunique(),
            "early_alarm_rate": round(float(early.notna().mean()), 3),
            "near_failure_units": len(lead),
            "onset_rul": onset,
        }
        | _lead_stats(lead)
        | false_alarms(df, thr, k, onset)
        | _lead_stats(lead_times(df, thr, k, onset=None), prefix="legacy_")
        | {f"auroc_{lo}_{hi}": round(auroc(df, lo, hi), 3) for lo, hi in AUROC_BANDS}
    )


def onset_sensitivity(df: pd.DataFrame, thr: float, k: int, onsets=ONSET_RUL_SENSITIVITY) -> list:
    """Detection, lead and false alarms for several onset proxies (same thr and k)."""
    return [
        {"onset_rul": r} | _lead_stats(lead_times(df, thr, k, r)) | false_alarms(df, thr, k, r)
        for r in onsets
    ]


def select_k(val_df: pd.DataFrame, thr: float, candidates=ALARM_K_CANDIDATES) -> int:
    """Pre-registered rule: smallest k with no early alarm on val; else the largest k."""
    for k in candidates:
        if evaluate(val_df, thr, k)["early_alarm_rate"] == 0:
            return k
    return max(candidates)


def alarm_state(df: pd.DataFrame, thr: float, k: int) -> pd.Series:
    """True while the alarm is on: the current streak above thr has reached k."""
    return run_length(df, thr) >= k


def alarm_onsets(df: pd.DataFrame, thr: float, k: int) -> pd.Series:
    """True only at the cycle where a streak reaches k: one event per streak."""
    return run_length(df, thr) == k


def lead_times(df: pd.DataFrame, thr: float, k: int, onset: int | None = ONSET_RUL) -> pd.Series:
    """RUL at the first cycle after the healthy window with the alarm on, per near-failure unit.

    onset=None: legacy definition, any alarm after the healthy window counts.
    onset=R: only cycles with RUL <= R count; earlier alarms are false alarms (false_alarms).
    Indexed by near-failure unit; NaN where the unit is not detected.
    """
    df = df.sort_values(["unit", "cycle"])
    end_rul = df.groupby("unit")["rul"].min()
    near = end_rul[end_rul <= NEAR_FAILURE_RUL].index
    on = alarm_state(df, thr, k) & (df["cycle"] > HEALTHY_CYCLES)
    if onset is not None:
        on &= df["rul"] <= onset
    return df[on].groupby("unit")["rul"].first().reindex(near)  # sorted: first = earliest


def false_alarms(df: pd.DataFrame, thr: float, k: int, onset: int = ONSET_RUL) -> dict:
    """Alarm events in mid-life (cycle > healthy window, RUL > onset)."""
    df = df.sort_values(["unit", "cycle"])
    mid = (df["cycle"] > HEALTHY_CYCLES) & (df["rul"] > onset)
    events = alarm_onsets(df, thr, k) & mid
    n_units, n_cycles = df.loc[mid, "unit"].nunique(), int(mid.sum())
    return {
        "mid_units": n_units,
        "fa_engine_rate": round(df.loc[events, "unit"].nunique() / n_units, 3)
        if n_units
        else np.nan,
        "fa_per_1000": round(1000 * int(events.sum()) / n_cycles, 3) if n_cycles else np.nan,
    }


def _lead_stats(lead: pd.Series, prefix: str = "") -> dict:
    """lead is indexed by near-failure unit (NaN = not detected)."""
    det = lead.dropna()
    has = len(det) > 0
    return {
        f"{prefix}detection_rate": round(len(det) / max(len(lead), 1), 3),
        f"{prefix}lead_median": float(det.median()) if has else np.nan,
        f"{prefix}lead_q25": float(det.quantile(0.25)) if has else np.nan,
        f"{prefix}lead_q75": float(det.quantile(0.75)) if has else np.nan,
    }
