"""Load MBTA Bus Arrival/Departure CSVs into the stop_events table.

    python -m src.load_historical                  # the three most recent monthly CSVs
    python -m src.load_historical --months 6       # the six most recent
    python -m src.load_historical path/a.csv ...   # specific files

Safe to rerun: the table is unique on (service_date, route_id, direction_id, stop_id,
scheduled) and rows are inserted with ON CONFLICT DO NOTHING, so the first row for a
key wins and reruns insert nothing.
"""

import argparse
import io
import json
from pathlib import Path

import pandas as pd
import psycopg

from src.db import ROOT, database_url

RAW_DIR = ROOT / "data" / "raw"
SCHEMA = ROOT / "src" / "schema.sql"
LOG_FILE = ROOT / "logs" / "load_historical.jsonl"

DUMMY_DATE = pd.Timestamp("1900-01-01")
MIN_DELAY_MIN, MAX_DELAY_MIN = -30, 120  # outside this range is a data error
CHUNK_ROWS = 500_000

KEY = ["service_date", "route_id", "direction_id", "stop_id", "scheduled"]
COLUMNS = ["source", *KEY, "actual", "half_trip_id", "time_point_id", "time_point_order",
           "point_type", "standard_type", "scheduled_headway", "headway"]

STAGING_DDL = """
CREATE TEMP TABLE IF NOT EXISTS staging (
    source text, service_date date, route_id text, direction_id text, stop_id text,
    scheduled timestamp, actual timestamp, half_trip_id text, time_point_id text,
    time_point_order smallint, point_type text, standard_type text,
    scheduled_headway integer, headway integer
)
"""

_cols = ", ".join(COLUMNS)
INSERT = f"""
INSERT INTO stop_events ({_cols})
SELECT {_cols} FROM staging
ON CONFLICT ON CONSTRAINT stop_events_key DO NOTHING
"""


def clock_to_timestamp(service_date, clock):
    """'1900-01-02T00:45:00Z' on 2026-09-15 -> 2026-09-16 00:45 (Boston wall clock).

    The CSV stores times as an offset from a dummy 1900-01-01 (the "Z" is not UTC),
    so trips after midnight roll into 1900-01-02.
    """
    return service_date + (pd.to_datetime(clock, format="%Y-%m-%dT%H:%M:%SZ") - DUMMY_DATE)


def prepare(raw):
    """Raw CSV chunk (all columns as strings) -> (rows ready for stop_events, counts)."""
    df = raw.copy()
    df["source"] = "historical"
    df["service_date"] = pd.to_datetime(df["service_date"], format="%Y-%m-%d")
    df["scheduled"] = clock_to_timestamp(df["service_date"], df["scheduled"])
    df["actual"] = clock_to_timestamp(df["service_date"], df["actual"])

    bad_key = df[KEY].isna().any(axis=1)
    delay_min = (df["actual"] - df["scheduled"]).dt.total_seconds() / 60
    out_of_range = delay_min.notna() & ~delay_min.between(MIN_DELAY_MIN, MAX_DELAY_MIN)
    df = df[~bad_key & ~out_of_range]
    duplicate = df.duplicated(subset=KEY)  # first row wins, as ON CONFLICT DO NOTHING would
    df = df[~duplicate]

    counts = {
        "read": len(raw),
        "bad_key": int(bad_key.sum()),
        "out_of_range": int((out_of_range & ~bad_key).sum()),
        "duplicate_key": int(duplicate.sum()),
        "no_actual": int(df["actual"].isna().sum()),
    }
    df["service_date"] = df["service_date"].dt.date
    return df[COLUMNS], counts


def latest_months(n):
    """The n most recent monthly CSVs under data/raw/, which must be consecutive."""
    files = sorted(RAW_DIR.rglob("MBTA-Bus-Arrival-Departure-Times_*.csv"), key=lambda p: p.stem[-7:])
    files = files[-n:]
    periods = [pd.Period(p.stem[-7:], freq="M") for p in files]
    if any((b - a).n != 1 for a, b in zip(periods, periods[1:])):
        raise SystemExit(f"months are not consecutive: {[str(p) for p in periods]}")
    return files


def load_file(conn, path):
    totals = {"read": 0, "bad_key": 0, "out_of_range": 0, "duplicate_key": 0, "no_actual": 0, "inserted": 0}
    for raw in pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], chunksize=CHUNK_ROWS):
        rows, counts = prepare(raw)
        for k, v in counts.items():
            totals[k] += v

        buf = io.StringIO()
        rows.to_csv(buf, index=False, header=False)
        with conn.cursor() as cur:
            cur.execute("TRUNCATE staging")
            with cur.copy(f"COPY staging ({_cols}) FROM STDIN WITH (FORMAT csv)") as copy:
                copy.write(buf.getvalue())
            cur.execute(INSERT)
            totals["inserted"] += cur.rowcount
        conn.commit()
    return totals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", type=Path, help="CSV files (default: the most recent --months)")
    parser.add_argument("--months", type=int, default=3, help="how many recent months to load (default 3)")
    args = parser.parse_args(argv)
    files = args.files or latest_months(args.months)
    if not files:
        parser.error(f"no CSV files found under {RAW_DIR}")

    LOG_FILE.parent.mkdir(exist_ok=True)
    with psycopg.connect(database_url()) as conn:
        conn.execute(SCHEMA.read_text())
        conn.execute(STAGING_DDL)
        conn.commit()
        for path in files:
            totals = load_file(conn, path)
            print(f"{path.name}: " + ", ".join(f"{k}={v:,}" for k, v in totals.items()), flush=True)
            with LOG_FILE.open("a") as log:
                log.write(json.dumps({"file": path.name, **totals}) + "\n")
        total = conn.execute("SELECT count(*) FROM stop_events WHERE source = 'historical'").fetchone()[0]
        print(f"stop_events historical rows: {total:,}")


if __name__ == "__main__":
    main()
