from datetime import datetime

from src.pull_daily import api_route_ids, map_record

API_ROUTES = {"1": "1", "9": "9", "66": "66", "741": "SL1", "708": "CT3"}


def schedule(direction=1, stop="64", departure="2026-10-29T14:15:00-04:00", arrival=None):
    return {
        "id": "schedule-1",
        "attributes": {"direction_id": direction, "departure_time": departure,
                       "arrival_time": arrival, "stop_sequence": 3},
        "relationships": {"stop": {"data": {"id": stop}}, "trip": {"data": {"id": "T1"}}},
    }


def test_zero_padded_and_named_routes_map_to_api_ids():
    mapping, unknown = api_route_ids({"01", "09", "66", "SL1", "CT3", "XX9"}, API_ROUTES)
    assert mapping == {"01": "1", "09": "9", "66": "66", "SL1": "741", "CT3": "708"}
    assert unknown == ["XX9"]


def test_record_maps_to_file_conventions():
    row = map_record(schedule(), "01", "2026-10-29")
    assert row["route_id"] == "01"
    assert row["direction_id"] == "Inbound"
    assert row["stop_id"] == "64"
    assert row["source"] == "api"
    assert datetime.fromisoformat(row["scheduled"]) == datetime(2026, 10, 29, 14, 15)  # offset dropped


def test_direction_zero_is_outbound():
    assert map_record(schedule(direction=0), "01", "2026-10-29")["direction_id"] == "Outbound"


def test_last_stop_falls_back_to_arrival_time():
    row = map_record(schedule(departure=None, arrival="2026-10-30T00:40:00-04:00"), "01", "2026-10-29")
    assert datetime.fromisoformat(row["scheduled"]) == datetime(2026, 10, 30, 0, 40)
    assert row["service_date"] == "2026-10-29"


def test_record_without_any_time_is_skipped():
    assert map_record(schedule(departure=None, arrival=None), "01", "2026-10-29") is None
