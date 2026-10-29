import pandas as pd
import pytest

import src.load_historical as lh
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


def test_route_and_direction_keep_file_conventions():
    rows, _ = run(raw_row(route="01"), raw_row(route="SL1", direction="Outbound", stop="2"))
    assert rows["route_id"].tolist() == ["01", "SL1"]
    assert rows["direction_id"].tolist() == ["Inbound", "Outbound"]


def test_times_become_wall_clock_timestamps_on_the_service_date():
    rows, _ = run(raw_row())
    assert rows["scheduled"].iloc[0] == pd.Timestamp("2026-09-08 14:15")
    assert rows["scheduled"].dt.tz is None


def test_past_midnight_rolls_into_next_calendar_day():
    rows, _ = run(raw_row(scheduled="1900-01-02T00:45:00Z", actual="1900-01-02T00:55:00Z"))
    assert rows["scheduled"].iloc[0] == pd.Timestamp("2026-09-09 00:45")
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


def test_duplicate_key_keeps_the_first_row():
    rows, counts = run(raw_row(), raw_row(actual="1900-01-01T14:30:00Z"))
    assert len(rows) == 1 and counts["duplicate_key"] == 1
    assert rows["actual"].iloc[0] == pd.Timestamp("2026-09-08 14:20")


def months_dir(tmp_path, monkeypatch, months):
    for m in months:
        (tmp_path / f"MBTA-Bus-Arrival-Departure-Times_{m}.csv").touch()
    monkeypatch.setattr(lh, "RAW_DIR", tmp_path)


def test_latest_months_picks_the_most_recent_consecutive_files(tmp_path, monkeypatch):
    months_dir(tmp_path, monkeypatch, ["2025-12", "2026-01", "2026-02", "2026-03"])
    assert [p.stem[-7:] for p in lh.latest_months(3)] == ["2026-01", "2026-02", "2026-03"]


def test_latest_months_rejects_a_gap(tmp_path, monkeypatch):
    months_dir(tmp_path, monkeypatch, ["2026-01", "2026-03", "2026-04"])
    with pytest.raises(SystemExit):
        lh.latest_months(3)
