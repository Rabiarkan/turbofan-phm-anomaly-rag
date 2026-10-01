import pandas as pd

from src.config import FEATURE_SENSORS
from src.features import healthy


def test_healthy_keeps_first_n_cycles_per_unit():
    df = pd.DataFrame({"unit": [1] * 5 + [2] * 3, "cycle": [1, 2, 3, 4, 5, 1, 2, 3]})
    out = healthy(df, n_cycles=2)
    assert out.groupby("unit")["cycle"].max().tolist() == [2, 2]
    assert len(out) == 4


def test_feature_sensors_are_14_unique_and_exclude_dropped():
    assert len(FEATURE_SENSORS) == len(set(FEATURE_SENSORS)) == 14
    assert not {"T2", "P2", "epr", "farB", "Nf_dmd", "PCNfR_dmd", "P15"} & set(FEATURE_SENSORS)
