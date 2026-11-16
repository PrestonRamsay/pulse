import numpy as np
import pandas as pd
import pytest

from src.features import add_day_type, add_history_features, window_counts


def rows(spec):
    """spec: list of (service_date, stop_id, late)."""
    df = pd.DataFrame(spec, columns=["service_date", "stop_id", "late"])
    df["service_date"] = pd.to_datetime(df["service_date"])
    df["route_id"], df["direction_id"], df["hour"] = "01", "Inbound", 8
    return add_day_type(df)


def test_day_type_follows_service_pattern_not_weekday():
    df = add_day_type(pd.DataFrame({"service_date": pd.to_datetime(
        ["2026-09-07", "2026-09-08", "2026-09-12", "2026-09-13", "2026-07-03"])}))
    assert df["day_type"].tolist() == ["sunday", "weekday", "saturday", "sunday", "saturday"]


def test_window_excludes_own_day_and_later():
    hist = rows([("2026-07-01", 1, 1), ("2026-07-02", 1, 0), ("2026-07-03", 1, 1), ("2026-07-03", 1, 1)])
    target = rows([("2026-07-03", 1, 0)])
    n, k = window_counts(hist, target, ["stop_id"], [7])[7]
    assert (n[0], k[0]) == (2, 1)  # 1 and 2 July only; both 3 July rows ignored


def test_window_length_is_in_days():
    hist = rows([("2026-07-01", 1, 1), ("2026-07-05", 1, 0)])
    target = rows([("2026-07-08", 1, 0)])
    n, _ = window_counts(hist, target, ["stop_id"], [3])[3]  # 5..7 July
    assert n[0] == 1


def test_keys_do_not_mix():
    hist = rows([("2026-07-01", 1, 1), ("2026-07-01", 2, 0)])
    target = rows([("2026-07-02", 2, 1)])
    n, k = window_counts(hist, target, ["stop_id"], [7])[7]
    assert (n[0], k[0]) == (1, 0)


def test_changing_a_label_never_moves_its_own_or_earlier_features():
    base = rows([(f"2026-07-{d:02d}", s, (d + s) % 2) for d in range(1, 15) for s in (1, 2)])
    flipped = base.copy()
    flipped.loc[flipped["service_date"] == "2026-07-10", "late"] ^= 1
    a = add_history_features(base, base, windows=(7,))
    b = add_history_features(flipped, flipped, windows=(7,))
    cols = [c for c in a.columns if c.startswith(("rate_", "n_"))]
    upto = base["service_date"] <= "2026-07-10"
    np.testing.assert_array_equal(a.loc[upto, cols].to_numpy(), b.loc[upto, cols].to_numpy())
    assert not np.array_equal(a.loc[~upto, cols].to_numpy(), b.loc[~upto, cols].to_numpy())


def test_rates_shrink_toward_parent_and_start_at_half():
    hist = rows([("2026-07-01", 1, 1)])
    target = rows([("2026-07-01", 1, 0), ("2026-07-02", 1, 0)])
    out = add_history_features(hist, target, windows=(7,), smoothing=1.0)
    assert out["rate_slot_7d"].iloc[0] == pytest.approx(0.5)  # no history at all yet
    assert 0.5 < out["rate_slot_7d"].iloc[1] < 1.0
