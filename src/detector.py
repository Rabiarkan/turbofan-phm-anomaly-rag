"""Frozen detector for Part 2: the 2-component Hotelling T² (README main result).

python -m src.detector   # fit, check against the results table, save, report val false alarms
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler

from src.baselines import T2
from src.config import FEATURE_SENSORS, HEALTHY_CYCLES, MODELS, PCA_TUNED_COMPONENTS, SUBSET
from src.evaluate import false_alarms, first_alarm_cycle, select_k, threshold
from src.features import prepare
from src.views import cycle_view

DEFAULT_PATH = MODELS / f"detector_{SUBSET}.joblib"


@dataclass
class Detector:
    scaler: StandardScaler  # fitted on healthy cycles of the fit engines
    model: T2  # 2-component PCA inside
    thr: float  # 99th percentile of val-healthy scores
    k: int  # consecutive exceedances, chosen on val
    features: list[str] = field(default_factory=lambda: list(FEATURE_SENSORS))
    sklearn_version: str = sklearn.__version__

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Raw sensor rows (physical names, unscaled) -> unit, cycle, score."""
        X = self.scaler.transform(df[self.features])
        return df[["unit", "cycle"]].assign(score=self.model.score(X))

    def first_alarms(self, df: pd.DataFrame) -> pd.Series:
        """First alarm cycle after the healthy window, per unit (NaN = no alarm). Uses no RUL."""
        return first_alarm_cycle(self.score(df), self.thr, self.k, after=HEALTHY_CYCLES)


def build(subset: str = SUBSET) -> tuple[Detector, dict]:
    """Same steps as the Part 1 runner for pca_t2_tuned; also returns val mid-life false alarms."""
    p = prepare(subset)
    X_fit, _ = cycle_view(p.fit_healthy)
    X_vh, _ = cycle_view(p.val_healthy)
    X_val, idx_val = cycle_view(p.val)
    model = T2(n_components=PCA_TUNED_COMPONENTS).fit(X_fit)
    thr = threshold(pd.Series(model.score(X_vh)))
    val = idx_val.assign(score=model.score(X_val))
    k = select_k(val, thr)
    return Detector(p.scaler, model, thr, k), false_alarms(val, thr, k)


def save(det: Detector, path: Path = DEFAULT_PATH) -> None:
    joblib.dump(vars(det), path)


def load(path: Path = DEFAULT_PATH) -> Detector:
    state = joblib.load(path)
    if state["sklearn_version"] != sklearn.__version__:
        warnings.warn(
            f"detector saved with scikit-learn {state['sklearn_version']}, "
            f"running {sklearn.__version__}; rebuild with python -m src.detector",
            stacklevel=2,
        )
    return Detector(**state)


if __name__ == "__main__":
    det, fa = build()
    # regression gate: must reproduce pca_t2_tuned in results/*_results_FD001.csv
    assert (round(det.thr, 4), det.k) == (5.7303, 3), (det.thr, det.k)
    save(det)
    print(f"saved {DEFAULT_PATH.name}: thr={det.thr:.4f}, k={det.k}")
    print(f"val mid-life false alarms (RUL > 125): {fa}")
