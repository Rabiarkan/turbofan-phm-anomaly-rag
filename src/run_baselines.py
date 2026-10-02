"""Fit every baseline on healthy fit cycles and evaluate on val

python -m src.run_baselines            # FD001
"""

import sys

import pandas as pd

from src.baselines import BASELINES
from src.config import FEATURE_SENSORS, RESULTS, SUBSET
from src.evaluate import evaluate, select_k, threshold
from src.features import prepare


def scored(df: pd.DataFrame, model) -> pd.DataFrame:
    out = df[["unit", "cycle", "rul"]].copy()
    out["score"] = model.score(df[FEATURE_SENSORS].to_numpy())
    return out


def main(subset: str = SUBSET) -> pd.DataFrame:
    p = prepare(subset)
    rows = []
    for name, cls in BASELINES.items():
        model = cls().fit(p.fit_healthy[FEATURE_SENSORS].to_numpy())
        thr = threshold(scored(p.val_healthy, model)["score"])
        val = scored(p.val, model)
        k = select_k(val, thr)
        rows.append({"model": name} | evaluate(val, thr, k))
    table = pd.DataFrame(rows).set_index("model")
    path = RESULTS / f"val_baselines_{subset}.csv"
    table.to_csv(path)
    print(table.to_string())
    print(f"written {path.relative_to(RESULTS.parent)}")
    return table


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else SUBSET)
