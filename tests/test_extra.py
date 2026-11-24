"""extra_data: earlier months either fill the history windows only, or are also trained on."""

import pandas as pd
import pytest

from src.extra import extra_md5s, extra_spec, keep_train_rows
from src.features import add_history_features


def frame(dates, late):
    return pd.DataFrame({"service_date": pd.to_datetime(dates), "late": late,
                         "route_id": "1", "direction_id": "0", "stop_id": "s", "hour": 8,
                         "day_type": "weekday"})


TRAIN_START = pd.Timestamp("2026-07-01")
ROWS = frame(["2026-06-29", "2026-06-30", "2026-07-01", "2026-07-02"], [1, 1, 0, 0])


def test_history_use_keeps_only_train_rows():
    kept = keep_train_rows(ROWS, {"extra_data": {"files": ["apr_jun"], "use": "history"}}, TRAIN_START)
    assert kept["service_date"].min() == TRAIN_START
    assert len(kept) == 2


def test_train_use_and_no_extra_keep_every_row():
    assert len(keep_train_rows(ROWS, {"extra_data": {"files": ["apr_jun"], "use": "train"}}, TRAIN_START)) == 4
    assert len(keep_train_rows(ROWS, {}, TRAIN_START)) == 4


def test_unknown_use_is_rejected():
    with pytest.raises(ValueError):
        extra_spec({"extra_data": {"files": ["apr_jun"], "use": "both"}})


def test_extra_months_reach_the_windows_but_never_later_days():
    """1 July sees the two late June days; 2 July also sees 1 July (on time); no day sees itself."""
    out = add_history_features(ROWS, ROWS, windows=(7,), smoothing=0.0)
    rate = dict(zip(out["service_date"].dt.strftime("%m-%d"), out["rate_network_7d"]))
    assert rate["07-01"] == pytest.approx(1.0)       # June 29-30: 2 of 2 late
    assert rate["07-02"] == pytest.approx(2 / 3)     # June 29-30 + July 1: 2 of 3 late
    assert out.loc[out["service_date"] == "2026-06-29", "n_network_7d"].iloc[0] == 0


def test_md5s_logged_per_extra_file():
    cfg = {"extra_data": {"files": ["apr_jun"], "use": "history"}}
    assert extra_md5s(cfg, lambda n: f"md5-of-{n}") == {"dvc_md5_apr_jun": "md5-of-apr_jun"}
    assert extra_md5s({}, lambda n: "x") == {}
