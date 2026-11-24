"""MBTA LAMP bus events: read them and convert to stop_events conventions.

LAMP (https://performancedata.mbta.com, code at github.com/mbta/lamp) publishes bus events
with TransitMaster actual times, refreshed daily:

  LAMP_RECENT_Bus_Events.parquet   ~175 MB, the last 7 service days (includes yesterday)
  LAMP_ALL_Bus_Events.parquet      ~8.8 GB, 31 Dec 2025 onward; read remotely by row group

Actual times exist only at timepoints, the same stops as the monthly Arrival/Departure files.
Checked against those files for 1-15 Sep 2026 (src/lamp_check.py): 95% of actual times are
identical to the second and the late label agrees on 98.8% of matched departures.

Conventions, found by matching LAMP to the monthly files:
* Times: use the seconds-after-midnight columns (plan_stop_departure_sam,
  tm_actual_departure_time_sam) on the service date. They are Boston wall clock in every file.
  The plan_stop_departure_dt column is shifted early by the UTC offset, so it is not used.
* Route ids: single digits are zero-padded in the files ('1' -> '01'); the Silver Line,
  CT3 and route 194 use GTFS ids in LAMP (ROUTE_MAP). School variants 191-193 and '746_'
  appear under their main route in LAMP (15, 39, 57, 746), so they load as that route.
* Direction: 0 = Outbound, 1 = Inbound (as in the V3 API).
* point_type: start/mid/end -> Startpoint/Midpoint/Endpoint; timepoint_abbr = time_point_id.
* Actual time: the departure at start and mid timepoints, the arrival at endpoints, as in the
  monthly files.
"""

import datetime as dt

import fsspec
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.db import ROOT

BASE_URL = "https://performancedata.mbta.com/lamp/tableau/bus"
RECENT_URL = f"{BASE_URL}/LAMP_RECENT_Bus_Events.parquet"
ALL_URL = f"{BASE_URL}/LAMP_ALL_Bus_Events.parquet"
LAMP_DIR = ROOT / "data" / "external" / "lamp"

# LAMP (GTFS) route id -> monthly-file route id, where they differ beyond zero-padding.
# Each pair matched 99.9-100% of that route's departures on 1-15 Sep 2026.
ROUTE_MAP = {"741": "SL1", "742": "SL2", "743": "SL3", "751": "SL4", "749": "SL5",
             "746": "SLW", "708": "CT3", "8993": "194"}
DIRECTION = {0: "Outbound", 1: "Inbound"}
POINT_TYPE = {"start": "Startpoint", "mid": "Midpoint", "end": "Endpoint"}
MIN_DELAY_MIN, MAX_DELAY_MIN = -30, 120  # same bad-record rule as src/load_historical.py

COLUMNS = ["service_date", "route_id", "direction_id", "stop_id", "trip_id", "timepoint_abbr",
           "timepoint_order", "point_type", "plan_stop_departure_sam", "tm_actual_departure_time_sam",
           "tm_actual_arrival_time_sam"]
KEY = ["service_date", "route_id", "direction_id", "stop_id", "scheduled"]


def file_route(route):
    """LAMP route id -> the monthly files' style."""
    route = str(route)
    if route in ROUTE_MAP:
        return ROUTE_MAP[route]
    return route.zfill(2) if route.isdigit() and len(route) == 1 else route


def read_all_range(start, end, url=ALL_URL):
    """Rows of the big LAMP file between two service dates, fetching only the row groups and
    columns needed (about 50 MB per month instead of the 8.8 GB file)."""
    with fsspec.filesystem("http").open(url, "rb", block_size=1 << 22, cache_type="readahead") as f:
        pf = pq.ParquetFile(f)
        md = pf.metadata
        sd = [md.schema.column(i).name for i in range(md.num_columns)].index("service_date")
        groups = [g for g in range(md.num_row_groups)
                  if md.row_group(g).column(sd).statistics.min <= end
                  and md.row_group(g).column(sd).statistics.max >= start]
        table = pf.read_row_groups(groups, columns=COLUMNS)
    keep = pc.and_(pc.greater_equal(table["service_date"], start), pc.less_equal(table["service_date"], end))
    return table.filter(keep).to_pandas()


def to_stop_events(raw):
    """LAMP rows -> stop_events rows (source 'lamp'), timepoints with an actual time only.
    Returns (rows, counts)."""
    # The monthly files record the departure at start and mid timepoints, and the arrival at
    # the end of the route (checked: LAMP's arrival matches them to the second 99.3% of the time there).
    end = raw["point_type"] == "end"
    actual_sam = raw["tm_actual_departure_time_sam"].where(~end, raw["tm_actual_arrival_time_sam"])
    df = raw.assign(actual_sam=actual_sam)
    df = df[df["actual_sam"].notna() & df["plan_stop_departure_sam"].notna()].copy()
    with_actual = len(df)
    df = df[~df["route_id"].astype(str).str.startswith("Shuttle-")]  # rail replacement shuttles
    midnight = pd.to_datetime(df["service_date"])
    df["scheduled"] = midnight + pd.to_timedelta(df["plan_stop_departure_sam"].astype("int64"), unit="s")
    df["actual"] = midnight + pd.to_timedelta(df["actual_sam"].astype("int64"), unit="s")
    df["service_date"] = midnight.dt.date
    df["route_id"] = df["route_id"].map(file_route)
    df["direction_id"] = df["direction_id"].map(DIRECTION)
    df["stop_id"] = df["stop_id"].astype(str)
    df["point_type"] = df["point_type"].map(POINT_TYPE)
    df["time_point_id"] = df["timepoint_abbr"]
    df["time_point_order"] = df["timepoint_order"].astype("Int16")
    df["trip_id"] = df["trip_id"].astype(str)

    bad_key = df[KEY].isna().any(axis=1)
    delay = (df["actual"] - df["scheduled"]).dt.total_seconds() / 60
    out_of_range = ~delay.between(MIN_DELAY_MIN, MAX_DELAY_MIN)
    df = df[~bad_key & ~out_of_range]
    duplicate = df.duplicated(subset=KEY)
    df = df[~duplicate]
    df["source"] = "lamp"
    counts = {"lamp_rows": len(raw), "with_actual": with_actual,
              "bad_key": int(bad_key.sum()), "out_of_range": int((out_of_range & ~bad_key).sum()),
              "duplicate_key": int(duplicate.sum()), "rows": len(df)}
    cols = ["source", *KEY, "actual", "trip_id", "time_point_id", "time_point_order", "point_type"]
    return df[cols].reset_index(drop=True), counts


def is_late(rows):
    return (rows["actual"] - rows["scheduled"]).dt.total_seconds() > 180


def complete_days(rows, today=None):
    """Drop today's (still running) service day; keep every finished one."""
    today = today or dt.date.today()
    return rows[rows["service_date"] < today]


def daily_summary(rows):
    out = rows.assign(late=is_late(rows)).groupby("service_date").agg(rows=("late", "size"), late_rate=("late", "mean"))
    return out.round({"late_rate": 4})
