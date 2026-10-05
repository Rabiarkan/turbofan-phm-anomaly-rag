"""
Make sure the windows aren't looking toward the future and that the engines aren't getting mixed up
"""

import numpy as np
import pandas as pd

from src.config import FEATURE_SENSORS
from src.views import rolling_view, window_view


def _df(n_cycles=(5, 3)):
    rows = []
    for u, n in enumerate(n_cycles, start=1):
        for c in range(1, n + 1):
            rows.append(
                {"unit": u, "cycle": c, "rul": n - c} | {s: 10 * u + c for s in FEATURE_SENSORS}
            )
    return pd.DataFrame(rows)


def test_window_view_is_causal_and_flattened():
    X, idx = window_view(_df(), w=3)
    # unit 1 (5 cycles) -> windows ending at 3, 4, 5; unit 2 (3 cycles) -> ending at 3
    assert idx[["unit", "cycle"]].values.tolist() == [[1, 3], [1, 4], [1, 5], [2, 3]]
    assert X.shape == (4, 3 * len(FEATURE_SENSORS))
    first_sensor = X[0, :: len(FEATURE_SENSORS)]  # same sensor at the 3 time steps
    assert first_sensor.tolist() == [11, 12, 13]  # cycles 1..3 of unit 1, never the future


def test_rolling_view_does_not_mix_units():
    X, idx = rolling_view(_df(), w=3)
    assert idx[["unit", "cycle"]].values.tolist() == [[1, 3], [1, 4], [1, 5], [2, 3]]
    assert np.allclose(X[:, 0], [12, 13, 14, 22])  # means of (11,12,13), (12,13,14), ... (21,22,23)
