"""Load MBTA Bus Arrival/Departure CSVs into the stop_events table.

    python -m src.load_historical                  # every CSV in data/raw/MBTA_Bus_Arrival_Departure_Times_2026/
    python -m src.load_historical path/a.csv ...   # specific files

Safe to rerun: rows already loaded are skipped. A historical row replaces an existing
'api' row for the same stop event, because only the historical feed has the actual time.
"""

import argparse
import io
import os
from pathlib import Path

import pandas as pd
import psycopg

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "MBTA_Bus_Arrival_Departure_Times_2026"
DEFAULT_DATABASE_URL = "postgresql://pulse:pulse@localhost:5432/pulse"

TIMEZONE = "America/New_York"  # CSV times say "Z" but are Boston local clock times
DUMMY_DATE = pd.Timestamp("1900-01-01")
DIRECTIONS = {"Outbound": 0, "Inbound": 1}  # matches the API's direction_id
MIN_DELAY_MIN, MAX_DELAY_MIN = -30, 120     # outside this range is a data error
CHUNK_ROWS = 500_000

KEY = ["service_date", "route_id", "direction_id", "stop_id", "scheduled"]
COLUMNS = ["source", *KEY, "actual", "half_trip_id", "time_point_id", "time_point_order",
           "point_type", "standard_type", "scheduled_headway", "headway"]

STAGING_DDL = """
CREATE TEMP TABLE staging (
    source text, service_date date, route_id text, direction_id smallint, stop_id text,
    scheduled timestamptz, actual timestamptz, half_trip_id text, time_point_id text,
    time_point_order smallint, point_type text, standard_type text,
    scheduled_headway integer, headway integer
)
"""

_cols = ", ".join(COLUMNS)
_updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in COLUMNS if c not in KEY)
UPSERT = f"""
INSERT INTO stop_events ({_cols})
SELECT {_cols} FROM staging
ON CONFLICT ON CONSTRAINT stop_events_key DO UPDATE
SET {_updates}, loaded_at = now()
WHERE stop_events.source = 'api'
"""


def clock_to_timestamp(service_date, clock):
    """'1900-01-02T00:45:00Z' on 2026-09-15 -> 2026-09-16 00:45 Boston time.

    The CSV stores times as an offset from a dummy 1900-01-01, so trips after
    midnight roll into 1900-01-02. Rows that fall in a DST gap/overlap become NaT.
    """
    offset = pd.to_datetime(clock, format="%Y-%m-%dT%H:%M:%SZ") - DUMMY_DATE
    local = service_date + offset
    return local.dt.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT")


def prepare(raw):
    """Raw CSV chunk (all columns as strings) -> (rows ready for stop_events, drop counts)."""
    df = raw.copy()
    df["source"] = "historical"
    df["service_date"] = pd.to_datetime(df["service_date"], format="%Y-%m-%d")
    df["route_id"] = df["route_id"].str.lstrip("0").replace("", "0")
    df["direction_id"] = df["direction_id"].map(DIRECTIONS).astype("Int16")
    df["scheduled"] = clock_to_timestamp(df["service_date"], df["scheduled"])
    df["actual"] = clock_to_timestamp(df["service_date"], df["actual"])

    bad_key = df[KEY].isna().any(axis=1)
    delay_min = (df["actual"] - df["scheduled"]).dt.total_seconds() / 60
    out_of_range = delay_min.notna() & ~delay_min.between(MIN_DELAY_MIN, MAX_DELAY_MIN)
    df = df[~bad_key & ~out_of_range]
    duplicate = df.duplicated(subset=KEY)
    df = df[~duplicate]

    counts = {
        "bad_key": int(bad_key.sum()),
        "out_of_range": int((out_of_range & ~bad_key).sum()),
        "duplicate_in_chunk": int(duplicate.sum()),
        "no_actual": int(df["actual"].isna().sum()),
    }
    df["service_date"] = df["service_date"].dt.date
    return df[COLUMNS], counts


def load_file(conn, path):
    totals = {"read": 0, "bad_key": 0, "out_of_range": 0, "duplicate_in_chunk": 0, "no_actual": 0, "written": 0}
    for raw in pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], chunksize=CHUNK_ROWS):
        rows, counts = prepare(raw)
        totals["read"] += len(raw)
        for k, v in counts.items():
            totals[k] += v

        buf = io.StringIO()
        rows.to_csv(buf, index=False, header=False)
        with conn.cursor() as cur:
            cur.execute("TRUNCATE staging")
            with cur.copy(f"COPY staging ({_cols}) FROM STDIN WITH (FORMAT csv)") as copy:
                copy.write(buf.getvalue())
            cur.execute(UPSERT)
            totals["written"] += cur.rowcount
        conn.commit()
    return totals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", type=Path, help="CSV files (default: all in the raw folder)")
    args = parser.parse_args(argv)
    files = args.files or sorted(RAW_DIR.glob("*.csv"))
    if not files:
        parser.error(f"no CSV files found in {RAW_DIR}")

    with psycopg.connect(os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)) as conn:
        conn.execute(STAGING_DDL)
        for path in files:
            totals = load_file(conn, path)
            print(f"{path.name}: " + ", ".join(f"{k}={v:,}" for k, v in totals.items()), flush=True)
        total = conn.execute("SELECT count(*) FROM stop_events WHERE source = 'historical'").fetchone()[0]
        print(f"stop_events historical rows: {total:,}")


if __name__ == "__main__":
    main()
