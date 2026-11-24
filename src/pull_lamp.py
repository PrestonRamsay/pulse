"""Load MBTA LAMP bus events (actual times) into stop_events with source = 'lamp'.

    python -m src.pull_lamp                                        # daily: the last 7 finished days
    python -m src.pull_lamp --start 2026-10-01 --end 2026-11-17    # backfill from the big file

Daily mode downloads LAMP_RECENT_Bus_Events.parquet (~175 MB, refreshed every morning) to
data/external/lamp/ and loads every finished service day in it (today is skipped while it
is still running). Backfill mode reads only the needed row groups of the big LAMP file,
one month at a time.

This is the "up to yesterday" label source the history features and the daily
recalibration need at serving time; the V3 API has no actual times.

Safe to rerun. For the same key, a monthly-file row ('historical') is never overwritten;
LAMP refreshes its own rows and fills in actual times on schedule-only 'api' rows. When a
monthly file arrives, src/load_historical.py replaces LAMP rows for its days. Training
splits read source = 'historical' only, so the frozen splits never change.
"""

import argparse
import datetime as dt
import io
import json

import fsspec
import pandas as pd
import psycopg

from src.db import ROOT, database_url
from src.lamp import LAMP_DIR, RECENT_URL, complete_days, daily_summary, read_all_range, to_stop_events

SCHEMA = ROOT / "src" / "schema.sql"
LOG_FILE = ROOT / "logs" / "pull_lamp.jsonl"
RECENT_FILE = LAMP_DIR / "LAMP_RECENT_Bus_Events.parquet"

COLUMNS = ["source", "service_date", "route_id", "direction_id", "stop_id", "scheduled", "actual",
           "trip_id", "time_point_id", "time_point_order", "point_type"]
_cols = ", ".join(COLUMNS)

STAGING_DDL = """
CREATE TEMP TABLE IF NOT EXISTS lamp_staging (
    source text, service_date date, route_id text, direction_id text, stop_id text,
    scheduled timestamp, actual timestamp, trip_id text, time_point_id text,
    time_point_order smallint, point_type text
)
"""

# historical > lamp > api: never touch a monthly-file row; refresh LAMP rows; upgrade API rows.
UPSERT = f"""
WITH up AS (
    INSERT INTO stop_events ({_cols})
    SELECT {_cols} FROM lamp_staging
    ON CONFLICT ON CONSTRAINT stop_events_key DO UPDATE SET
        source = 'lamp',
        actual = EXCLUDED.actual,
        trip_id = COALESCE(stop_events.trip_id, EXCLUDED.trip_id),
        time_point_id = COALESCE(stop_events.time_point_id, EXCLUDED.time_point_id),
        time_point_order = COALESCE(stop_events.time_point_order, EXCLUDED.time_point_order),
        point_type = COALESCE(stop_events.point_type, EXCLUDED.point_type),
        loaded_at = now()
    WHERE stop_events.source IN ('lamp', 'api')
    RETURNING (xmax = 0) AS inserted
)
SELECT count(*) FILTER (WHERE inserted), count(*) FILTER (WHERE NOT inserted) FROM up
"""


def download_recent(url=RECENT_URL, path=RECENT_FILE):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with fsspec.open(url, "rb") as src, tmp.open("wb") as dst:
        while block := src.read(1 << 22):
            dst.write(block)
    tmp.replace(path)
    return path


def upsert(conn, rows):
    """Write rows through a staging table. Returns (inserted, updated)."""
    buf = io.StringIO()
    rows[COLUMNS].to_csv(buf, index=False, header=False)
    with conn.cursor() as cur:
        cur.execute("TRUNCATE lamp_staging")
        with cur.copy(f"COPY lamp_staging ({_cols}) FROM STDIN WITH (FORMAT csv)") as copy:
            copy.write(buf.getvalue())
        inserted, updated = cur.execute(UPSERT).fetchone()
    conn.commit()
    return inserted, updated


def month_chunks(start, end):
    """(first, last) date pairs, one per calendar month, covering start..end."""
    day = start
    while day <= end:
        last = min((pd.Timestamp(day) + pd.offsets.MonthEnd(0)).date(), end)
        yield day, last
        day = last + dt.timedelta(days=1)


def load(conn, rows, counts, label):
    """Upsert finished days only (rows already passed through complete_days) and log the counts."""
    inserted, updated = upsert(conn, rows)
    days = daily_summary(rows)
    record = {"run_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "what": label,
              "first_day": str(days.index.min()), "last_day": str(days.index.max()), "days": len(days),
              **counts, "today_left_out": counts["rows"] - len(rows), "loaded": len(rows),
              "inserted": inserted, "updated": updated,
              "kept_monthly_file_rows": len(rows) - inserted - updated}
    print(f"{label}: {record['first_day']} to {record['last_day']} ({len(days)} days), "
          f"{len(rows):,} rows -> inserted {inserted:,}, updated {updated:,}, "
          f"kept monthly-file rows {record['kept_monthly_file_rows']:,} "
          f"(today, still running, left out: {record['today_left_out']:,})", flush=True)
    with LOG_FILE.open("a") as log:
        log.write(json.dumps(record) + "\n")
    return days


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=pd.Timestamp, help="backfill from this service date (big LAMP file)")
    parser.add_argument("--end", type=pd.Timestamp, help="backfill to this service date (default: yesterday)")
    args = parser.parse_args(argv)

    LOG_FILE.parent.mkdir(exist_ok=True)
    with psycopg.connect(database_url()) as conn:
        conn.execute(SCHEMA.read_text())
        conn.execute(STAGING_DDL)
        conn.commit()
        if args.start is None:
            raw = pd.read_parquet(download_recent())
            rows, counts = to_stop_events(raw)
            days = load(conn, complete_days(rows), counts, "recent")
        else:
            end = (args.end or pd.Timestamp(dt.date.today() - dt.timedelta(days=1))).date()
            for first, last in month_chunks(args.start.date(), end):
                rows, counts = to_stop_events(read_all_range(first, last))
                days = load(conn, complete_days(rows), counts, f"backfill {first:%Y-%m}")
        print(days.tail(7).to_string())
        n = conn.execute("SELECT source, count(*) FROM stop_events GROUP BY source ORDER BY source").fetchall()
        print("stop_events rows by source: " + ", ".join(f"{s} {c:,}" for s, c in n))


if __name__ == "__main__":
    main()
