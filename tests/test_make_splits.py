import pandas as pd
import pytest

from src.make_splits import add_calendar_features, check_order, split_by_date


def frame(rows):
    df = pd.DataFrame(rows, columns=["service_date", "scheduled"])
    return df.apply(pd.to_datetime)


def test_split_takes_two_earlier_months_then_halves_the_latest():
    df = frame([
        ("2026-06-30", "2026-06-30 08:00"),  # three months back: dropped
        ("2026-07-01", "2026-07-01 08:00"),
        ("2026-08-31", "2026-09-01 01:30"),  # after midnight, still August's service day
        ("2026-09-01", "2026-09-01 05:00"),
        ("2026-09-15", "2026-09-15 23:00"),
        ("2026-09-16", "2026-09-16 05:00"),
        ("2026-09-30", "2026-09-30 22:00"),
    ])
    train, val, test = split_by_date(df)
    assert train["service_date"].dt.strftime("%m-%d").tolist() == ["07-01", "08-31"]
    assert val["service_date"].dt.day.tolist() == [1, 15]
    assert test["service_date"].dt.day.tolist() == [16, 30]
    check_order(train, val, test)


def test_check_order_rejects_overlap():
    train = frame([("2026-08-31", "2026-09-01 06:00")])
    val = frame([("2026-09-01", "2026-09-01 05:00")])
    test = frame([("2026-09-16", "2026-09-16 05:00")])
    with pytest.raises(AssertionError):
        check_order(train, val, test)


def test_calendar_features_come_from_scheduled():
    df = add_calendar_features(frame([("2026-09-07", "2026-09-07 14:37")]))  # a Monday
    assert (df["day_of_week"].iloc[0], df["hour"].iloc[0], df["minute_of_day"].iloc[0]) == (0, 14, 877)
