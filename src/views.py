"""Ways to turn a scaled DataFrame into model inputs.

Each view returns (X, index) where index has unit, cycle, rul for every row of X.
Windowed views are causal: the row for cycle t only uses cycles t-W+1 .. t.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.config import FEATURE_SENSORS, WINDOW

INDEX = ["unit", "cycle", "rul"]


def cycle_view(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """One row per cycle (no time context)."""
    return df[FEATURE_SENSORS].to_numpy(), df[INDEX].reset_index(drop=True)


def window_view(df: pd.DataFrame, w: int = WINDOW) -> tuple[np.ndarray, pd.DataFrame]:
    """One row per window: the last w cycles flattened into w * n_sensors values."""
    xs, idx = [], []
    for _, g in df.sort_values(["unit", "cycle"]).groupby("unit"):
        a = g[FEATURE_SENSORS].to_numpy()
        if len(a) < w:
            continue
        win = sliding_window_view(a, (w, a.shape[1]))[:, 0]  # (n - w + 1, w, n_sensors)
        xs.append(win.reshape(len(win), -1))
        idx.append(g[INDEX].iloc[w - 1 :])
    return np.vstack(xs), pd.concat(idx).reset_index(drop=True)


def rolling_view(df: pd.DataFrame, w: int = WINDOW) -> tuple[np.ndarray, pd.DataFrame]:
    """One row per cycle: each sensor averaged over the last w cycles."""
    df = df.sort_values(["unit", "cycle"])
    rolled = df.groupby("unit")[FEATURE_SENSORS].transform(lambda s: s.rolling(w).mean())
    keep = rolled.notna().all(axis=1)
    return rolled[keep].to_numpy(), df.loc[keep, INDEX].reset_index(drop=True)
