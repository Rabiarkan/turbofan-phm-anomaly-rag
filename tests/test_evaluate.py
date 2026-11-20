import numpy as np
import pandas as pd

from src.evaluate import (
    false_alarms,
    first_alarm_cycle,
    lead_times,
    run_length,
    select_k,
)


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


def _life(alarm_cycles, n=200):
    """One unit run to failure (RUL n-1 .. 0); score 2 on alarm_cycles, else 0."""
    score = np.zeros(n)
    score[[c - 1 for c in alarm_cycles]] = 2
    return pd.DataFrame(
        {"unit": 1, "cycle": range(1, n + 1), "rul": range(n - 1, -1, -1), "score": score}
    )


# n=200: mid-life (cycle > 30, RUL > 125) is cycles 31..74, i.e. 44 cycles
HEALTHY_STREAK = list(range(28, 36))  # reaches k=2 at cycle 29, inside the healthy window
MID_STREAK = list(range(40, 45))  # reaches k=2 at cycle 41, in mid-life (RUL 159)
LATE_STREAK = list(range(180, 201))  # reaches k=2 at cycle 181 (RUL 19)


def test_false_alarm_counts_one_event_per_streak_and_only_in_mid_life():
    fa = false_alarms(_life(HEALTHY_STREAK + MID_STREAK), thr=1, k=2, onset=125)
    assert fa["mid_units"] == 1
    assert fa["fa_engine_rate"] == 1.0
    assert fa["fa_per_1000"] == round(1000 * 1 / 44, 3)  # one event, not 5 cycles, not 2 streaks


def test_detection_requires_rul_below_onset_but_legacy_does_not():
    df = _life(MID_STREAK)
    assert lead_times(df, thr=1, k=2, onset=None).loc[1] == 159  # legacy: mid-life alarm = lead
    assert pd.isna(lead_times(df, thr=1, k=2, onset=125).loc[1])  # new: not a detection
    df = _life(MID_STREAK + LATE_STREAK)
    assert lead_times(df, thr=1, k=2, onset=125).loc[1] == 19


def test_permanent_alarm_is_capped_at_onset_and_exposed_as_false_alarm():
    df = _life(range(31, 201))  # alarms from cycle 31 to failure
    assert lead_times(df, thr=1, k=2, onset=125).loc[1] == 125  # lead <= R
    assert false_alarms(df, thr=1, k=2, onset=125)["fa_engine_rate"] == 1.0
