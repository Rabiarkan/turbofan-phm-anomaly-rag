import numpy as np

from src.config import FEATURE_SENSORS
from src.lstm_ae import LSTMAE


def test_lstm_ae_scores_unseen_shift_higher():
    rng = np.random.default_rng(0)
    d, w = len(FEATURE_SENSORS), 5
    healthy = rng.normal(size=(300, w * d)).astype("float32")
    val = rng.normal(size=(60, w * d)).astype("float32")
    shifted = val + 4.0
    m = LSTMAE(seed=0).fit(healthy, val)
    assert m.score(shifted).mean() > m.score(val).mean()
    assert m.score(val).shape == (60,)
