import numpy as np
import pandas as pd
import pytest

from src.calibrate import apply_shifts, ece, level_shift, rolling_shifts, threshold_table


def test_level_shift_matches_target_rate():
    p = np.array([0.2, 0.4, 0.6])
    b = level_shift(p, 0.6)
    assert (1 / (1 + np.exp(-(np.log(p / (1 - p)) + b)))).mean() == pytest.approx(0.6)


def test_rolling_shift_uses_only_earlier_days():
    dates = pd.to_datetime(["2026-09-01"] * 2 + ["2026-09-02"] * 2 + ["2026-09-03"] * 2)
    p = np.full(6, 0.4)
    y = np.array([1, 1, 0, 1, 0, 0])
    shifts = rolling_shifts(dates, p, y, window=1)
    assert shifts[pd.Timestamp("2026-09-01")] == 0.0          # nothing before the first day
    assert shifts[pd.Timestamp("2026-09-02")] > 0             # 1 Sep ran 100% late
    out = apply_shifts(dates, p, shifts)
    assert out[4:].mean() == pytest.approx(0.5, abs=1e-6)     # 2 Sep ran 50%; 3 Sep's own labels unused


def test_shift_keeps_ranking():
    p = np.array([0.1, 0.5, 0.9])
    out = apply_shifts(pd.to_datetime(["2026-09-02"] * 3), p, {pd.Timestamp("2026-09-02"): 0.7})
    assert np.all(np.diff(out) > 0)


def test_ece_zero_when_calibrated():
    y = np.array([0, 1] * 50)
    assert ece(y, np.full(100, 0.5)) == pytest.approx(0.0)


def test_threshold_table_counts():
    t = threshold_table(np.array([1, 1, 0, 0]), np.array([0.9, 0.3, 0.6, 0.1]), thresholds=(0.5,))
    row = t.iloc[0]
    assert (row.warned_share, row.recall, row.precision, row.unwarned_late_rate) == (0.5, 0.5, 0.5, 0.5)
