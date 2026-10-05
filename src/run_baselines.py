"""Fit every model on healthy fit cycles and evaluate.

python -m src.run_baselines            # FD001, val only (development)
python -m src.run_baselines FD001 test # final run on the locked test set: do this once
"""

import sys
from functools import partial

import pandas as pd

from src.baselines import SPE, T2, Distance, IForest
from src.config import LSTM_SEEDS, PCA_TUNED_COMPONENTS, RESULTS, SUBSET
from src.evaluate import evaluate, select_k, threshold
from src.features import prepare
from src.lstm_ae import LSTMAE
from src.views import cycle_view, rolling_view, window_view

T2_TUNED = partial(T2, n_components=PCA_TUNED_COMPONENTS)

# name: (view, model factory). Rows marked "tuned" used val results to pick a setting.
EXPERIMENTS = {
    "distance": (cycle_view, Distance),
    "pca_spe": (cycle_view, SPE),
    "pca_t2": (cycle_view, T2),
    "iforest": (cycle_view, IForest),
    "pca_t2_tuned": (cycle_view, T2_TUNED),
    "win_pca_spe": (window_view, SPE),
    "win_pca_t2": (window_view, T2),
    "roll_distance": (rolling_view, Distance),
    "roll_pca_t2_tuned": (rolling_view, T2_TUNED),
} | {f"lstm_ae_s{s}": (window_view, partial(LSTMAE, seed=s)) for s in LSTM_SEEDS}

COLS = ["k", "early_alarm_rate", "detection_rate", "lead_median", "auroc_30_60", "auroc_60_90"]


def run(view, make, p, split: str) -> dict:
    X_fit, _ = view(p.fit_healthy)
    X_vh, _ = view(p.val_healthy)
    X_val, idx_val = view(p.val)
    model = make()
    model = model.fit(X_fit, X_vh) if getattr(model, "uses_val", False) else model.fit(X_fit)

    # threshold and k always come from val; test only receives the frozen choices
    thr = threshold(pd.Series(model.score(X_vh)))
    k = select_k(idx_val.assign(score=model.score(X_val)), thr)
    X_eval, idx_eval = (X_val, idx_val) if split == "val" else view(p.test)
    result = evaluate(idx_eval.assign(score=model.score(X_eval)), thr, k)
    return {"best_epoch": getattr(model, "best_epoch", None)} | result


def summarize_seeds(table: pd.DataFrame) -> pd.DataFrame:
    lstm = table[table.index.str.startswith("lstm_ae_s")]
    mean, std = lstm[COLS].mean(), lstm[COLS].std()
    table.loc["lstm_ae_mean"] = mean
    table.loc["lstm_ae_std"] = std
    return table


def main(subset: str = SUBSET, split: str = "val") -> pd.DataFrame:
    p = prepare(subset)
    rows = [{"model": n} | run(view, make, p, split) for n, (view, make) in EXPERIMENTS.items()]
    table = summarize_seeds(pd.DataFrame(rows).set_index("model"))
    path = RESULTS / f"{split}_results_{subset}.csv"
    table.to_csv(path)
    print(table[["best_epoch"] + COLS].round(3).to_string())
    print(f"written {path.relative_to(RESULTS.parent)}")
    return table


if __name__ == "__main__":
    args = sys.argv[1:] + [SUBSET, "val"][len(sys.argv[1:]) :]
    main(args[0], args[1])
