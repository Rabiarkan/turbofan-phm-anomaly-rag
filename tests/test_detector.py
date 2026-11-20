import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.baselines import T2
from src.config import FEATURE_SENSORS
from src.detector import Detector, load, save


def _engine(n=80, drift_from=None, seed=0, unit=1):
    """Raw sensor rows; optional linear drift on every sensor from cycle drift_from + 1."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURE_SENSORS)))
    if drift_from is not None:
        X[drift_from:] += np.linspace(0, 6, n - drift_from)[:, None]
    return pd.DataFrame(X, columns=FEATURE_SENSORS).assign(unit=unit, cycle=np.arange(1, n + 1))


def _detector():
    healthy = _engine(n=300)
    scaler = StandardScaler().fit(healthy[FEATURE_SENSORS])
    Z = scaler.transform(healthy[FEATURE_SENSORS])
    model = T2(n_components=0.99).fit(Z)
    return Detector(scaler, model, thr=float(np.quantile(model.score(Z), 0.99)), k=3)


def test_save_load_roundtrip_gives_identical_scores(tmp_path):
    det, eng = _detector(), _engine(drift_from=50, seed=1)
    path = tmp_path / "det.joblib"
    save(det, path)
    pd.testing.assert_frame_equal(load(path).score(eng), det.score(eng))


def test_first_alarm_uses_no_rul_and_fires_after_the_drift():
    det, eng = _detector(), _engine(drift_from=50, seed=1)
    assert "rul" not in eng.columns
    assert 50 < det.first_alarms(eng).loc[1] <= 80
