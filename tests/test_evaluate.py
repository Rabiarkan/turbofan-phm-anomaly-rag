import pandas as pd

from src.evaluate import first_alarm_cycle, run_length, select_k


def _unit(scores, unit=1):
    n = len(scores)
    return pd.DataFrame(
        {"unit": unit, "cycle": range(1, n + 1), "rul": range(n - 1, -1, -1), "score": scores}
    )


def test_run_length_resets_after_gap():
    df = _unit([0, 2, 2, 0, 2, 2, 2])
    assert run_length(df, thr=1).tolist() == [0, 1, 2, 0, 1, 2, 3]


def test_first_alarm_needs_k_consecutive():
    df = _unit([0, 2, 2, 0, 2, 2, 2])
    assert first_alarm_cycle(df, thr=1, k=2).loc[1] == 3
    assert first_alarm_cycle(df, thr=1, k=3).loc[1] == 7
    assert pd.isna(first_alarm_cycle(df, thr=1, k=4).loc[1])


def test_first_alarm_respects_after_and_units():
    df = pd.concat([_unit([2, 2, 0, 0, 2, 2], unit=1), _unit([0, 0, 0, 0, 0, 0], unit=2)])
    out = first_alarm_cycle(df, thr=1, k=2, after=3)
    assert out.loc[1] == 6
    assert pd.isna(out.loc[2])


def test_select_k_picks_smallest_k_without_early_alarm():
    # unit 1: two-cycle blip in the healthy window, then a long run near failure
    scores = [0] * 10 + [2, 2] + [0] * 28 + [2] * 10
    df = _unit(scores)
    assert select_k(df, thr=1, candidates=(1, 2, 3, 5)) == 3
