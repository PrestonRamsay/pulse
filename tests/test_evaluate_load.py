"""src.evaluate.load_all builds the scored splits like training does, extra months included.
Uses tiny made-up files in a temp folder; the real splits are never read."""

import pandas as pd
import pytest

import src.evaluate as ev
import src.extra as ex


def days(start, n, late):
    d = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame({"service_date": d, "late": late, "route_id": "1", "direction_id": "Inbound",
                         "stop_id": "s", "time_point_id": "t", "point_type": "Midpoint",
                         "standard_type": "Schedule", "hour": 8})


@pytest.fixture
def split_dir(tmp_path, monkeypatch):
    days("2026-06-01", 30, 1).to_parquet(tmp_path / "apr_jun.parquet")   # extra: all late
    days("2026-07-01", 20, 0).to_parquet(tmp_path / "train.parquet")     # all on time
    days("2026-07-21", 5, 0).to_parquet(tmp_path / "val.parquet")
    monkeypatch.setattr(ev, "SPLIT_DIR", tmp_path)
    monkeypatch.setattr(ex, "SPLIT_DIR", tmp_path)
    return tmp_path


CFG = {"history": {"windows": [7], "smoothing": 0.0}, "train": {"burn_in_days": 3}}


def test_extra_as_training_rows(split_dir):
    train, val = ev.load_all({**CFG, "extra_data": {"files": ["apr_jun"], "use": "train"}}, names=("train", "val"))
    assert train["service_date"].min() == pd.Timestamp("2026-06-04")     # extra start + 3-day burn-in
    assert train["service_date"].max() == pd.Timestamp("2026-07-20")
    assert list(val["service_date"].dt.day) == [21, 22, 23, 24, 25]      # val keeps exactly its own days


def test_extra_as_history_only(split_dir):
    train, val = ev.load_all({**CFG, "extra_data": {"files": ["apr_jun"], "use": "history"}}, names=("train", "val"))
    assert train["service_date"].min() == pd.Timestamp("2026-07-04")     # train start + burn-in, no June rows
    # 1 July's 7-day window is all June (late) because the extra months fed the windows
    first = train.iloc[0]
    assert first["service_date"] == pd.Timestamp("2026-07-04")
    assert first["rate_network_7d"] == pytest.approx(4 / 7)              # 27-30 June late, 1-3 July on time


def test_without_extra_matches_the_old_behaviour(split_dir):
    train, val = ev.load_all(CFG, names=("train", "val"))
    assert train["service_date"].min() == pd.Timestamp("2026-07-04")
    assert len(val) == 5
