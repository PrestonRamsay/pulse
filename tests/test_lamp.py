"""LAMP rows -> stop_events conventions (src/lamp.py), and the daily pull's date handling."""

import datetime as dt

import pandas as pd

from src.lamp import complete_days, file_route, to_stop_events
from src.pull_lamp import month_chunks

DAY = dt.date(2026, 9, 9)


def raw(**overrides):
    row = {"service_date": DAY, "route_id": "1", "direction_id": 0, "stop_id": "64", "trip_id": "78181030",
           "timepoint_abbr": "nubn", "timepoint_order": 1, "point_type": "start",
           "plan_stop_departure_sam": 16620,          # 04:37:00
           "tm_actual_departure_time_sam": 16739.0,   # 04:38:59
           "tm_actual_arrival_time_sam": 16700.0}
    row.update(overrides)
    return pd.DataFrame([row])


def test_route_ids_follow_the_monthly_files():
    assert file_route("1") == "01" and file_route("9") == "09"
    assert file_route("10") == "10" and file_route("111") == "111"
    assert file_route("741") == "SL1" and file_route("749") == "SL5" and file_route("708") == "CT3"


def test_times_come_from_seconds_after_midnight_on_the_service_date():
    rows, _ = to_stop_events(raw())
    r = rows.iloc[0]
    assert r["scheduled"] == pd.Timestamp("2026-09-09 04:37:00")
    assert r["actual"] == pd.Timestamp("2026-09-09 04:38:59")
    assert (r["source"], r["route_id"], r["direction_id"], r["point_type"]) == ("lamp", "01", "Outbound", "Startpoint")


def test_after_midnight_trips_stay_on_their_service_date():
    rows, _ = to_stop_events(raw(plan_stop_departure_sam=24 * 3600 + 600, tm_actual_departure_time_sam=24 * 3600 + 700.0))
    assert rows.iloc[0]["service_date"] == DAY
    assert rows.iloc[0]["scheduled"] == pd.Timestamp("2026-09-10 00:10:00")


def test_endpoints_use_the_arrival_like_the_monthly_files():
    rows, _ = to_stop_events(raw(point_type="end", tm_actual_departure_time_sam=17000.0))
    assert rows.iloc[0]["actual"] == pd.Timestamp("2026-09-09 04:38:20")  # arrival, 16700 s
    assert rows.iloc[0]["point_type"] == "Endpoint"


def test_rows_without_actuals_shuttles_and_bad_delays_are_dropped():
    frames = pd.concat([raw(), raw(stop_id="1", tm_actual_departure_time_sam=None),
                        raw(stop_id="2", route_id="Shuttle-BackBayKenmore"),
                        raw(stop_id="3", tm_actual_departure_time_sam=16620 + 3 * 3600.0)], ignore_index=True)
    rows, counts = to_stop_events(frames)
    assert list(rows["stop_id"]) == ["64"]
    assert counts["out_of_range"] == 1 and counts["with_actual"] == 3


def test_today_is_left_out_until_it_finishes():
    rows, _ = to_stop_events(pd.concat([raw(), raw(service_date=dt.date(2026, 9, 10))], ignore_index=True))
    assert list(complete_days(rows, today=dt.date(2026, 9, 10))["service_date"]) == [DAY]


def test_backfill_runs_one_calendar_month_at_a_time():
    chunks = list(month_chunks(dt.date(2026, 10, 15), dt.date(2026, 12, 2)))
    assert chunks == [(dt.date(2026, 10, 15), dt.date(2026, 10, 31)),
                      (dt.date(2026, 11, 1), dt.date(2026, 11, 30)),
                      (dt.date(2026, 12, 1), dt.date(2026, 12, 2))]
