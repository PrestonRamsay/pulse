import pandas as pd

from src.load_historical import prepare

HEADER = ["service_date", "route_id", "direction_id", "half_trip_id", "stop_id", "time_point_id",
          "time_point_order", "point_type", "standard_type", "scheduled", "actual",
          "scheduled_headway", "headway"]


def raw_row(route="01", direction="Inbound", stop="17861", scheduled="1900-01-01T14:15:00Z",
            actual="1900-01-01T14:20:00Z", date="2026-09-08"):
    return [date, route, direction, None, stop, "rugg", "1", "Endpoint", "Schedule",
            scheduled, actual, "900", None]


def run(*rows):
    return prepare(pd.DataFrame(list(rows), columns=HEADER))


def test_route_leading_zero_is_stripped_to_match_api():
    rows, _ = run(raw_row(route="01"), raw_row(route="SL1", stop="2"))
    assert rows["route_id"].tolist() == ["1", "SL1"]


def test_direction_text_becomes_api_direction_id():
    rows, _ = run(raw_row(direction="Outbound"), raw_row(direction="Inbound", stop="2"))
    assert rows["direction_id"].tolist() == [0, 1]


def test_times_become_boston_timestamps():
    rows, _ = run(raw_row())
    assert rows["scheduled"].iloc[0] == pd.Timestamp("2026-09-08 14:15", tz="America/New_York")


def test_past_midnight_rolls_into_next_calendar_day():
    rows, _ = run(raw_row(scheduled="1900-01-02T00:45:00Z", actual="1900-01-02T00:55:00Z"))
    assert rows["scheduled"].iloc[0] == pd.Timestamp("2026-09-09 00:45", tz="America/New_York")
    assert str(rows["service_date"].iloc[0]) == "2026-09-08"


def test_delays_outside_minus_30_to_120_are_dropped():
    rows, counts = run(
        raw_row(stop="a", actual="1900-01-01T13:44:00Z"),  # -31 min
        raw_row(stop="b", actual="1900-01-01T13:45:00Z"),  # -30 min, kept
        raw_row(stop="c", actual="1900-01-01T16:15:00Z"),  # +120 min, kept
        raw_row(stop="d", actual="1900-01-01T16:16:00Z"),  # +121 min
    )
    assert rows["stop_id"].tolist() == ["b", "c"]
    assert counts["out_of_range"] == 2


def test_missing_actual_is_kept_with_null():
    rows, counts = run(raw_row(actual=None))
    assert len(rows) == 1 and pd.isna(rows["actual"].iloc[0])
    assert counts["no_actual"] == 1


def test_duplicate_key_is_dropped():
    rows, counts = run(raw_row(), raw_row(actual="1900-01-01T14:30:00Z"))
    assert len(rows) == 1 and counts["duplicate_in_chunk"] == 1
