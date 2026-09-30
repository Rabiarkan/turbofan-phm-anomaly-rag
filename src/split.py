"""Split training engines into fit / validation sets by unit.

python -m src.split            # FD001
python -m src.split FD003
"""

import json
import sys

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, SEED, SPLITS, SUBSET, VAL_FRACTION


def split_units(units, val_fraction: float = VAL_FRACTION, seed: int = SEED) -> dict:
    units = np.sort(np.unique(units))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(units)
    n_val = round(len(units) * val_fraction)
    val = sorted(int(u) for u in shuffled[:n_val])
    fit = sorted(int(u) for u in shuffled[n_val:])
    assert not set(fit) & set(val), "a unit is in both sets"
    return {"seed": seed, "val_fraction": val_fraction, "fit": fit, "val": val}


def main(subset: str = SUBSET) -> None:
    train = pd.read_parquet(DATA_PROCESSED / f"train_{subset}.parquet")
    split = split_units(train["unit"])
    path = SPLITS / f"{subset}.json"
    path.write_text(json.dumps(split, indent=2))

    life = train.groupby("unit")["cycle"].max()
    for name in ("fit", "val"):
        q = life.loc[split[name]].quantile([0, 0.5, 1]).astype(int).tolist()
        print(f"{subset} {name}: {len(split[name])} units, life min/median/max {q}")
    print(f"written {path.relative_to(SPLITS.parents[1])}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else SUBSET)
