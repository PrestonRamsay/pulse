"""Pull today's planned bus service from the MBTA V3 API into stop_events.

    python -m src.pull_daily                     # today's Boston service date
    python -m src.pull_daily --date 2026-10-30
    python -m src.pull_daily --routes 01 66      # a few routes only (smoke test)

One /schedules request per route in the historical table, kept to the (route, direction,
stop) timepoints the historical files cover, inserted with source = 'api' and actual,
delay_min and late left NULL. Set MBTA_API_KEY to lift the 20 requests/minute limit.
Each run appends its counts to logs/pull_daily.log.
"""

import argparse
import io
import json
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import psycopg
import requests

from src.db import ROOT, database_url

API_BASE = "https://api-v3.mbta.com"
BOSTON = ZoneInfo("America/New_York")
LOG_FILE = ROOT / "logs" / "pull_daily.log"
DIRECTIONS = {0: "Outbound", 1: "Inbound"}
NO_KEY_INTERVAL_S = 3.1  # 20 requests a minute without a key
KEY_INTERVAL_S = 0.1

COLUMNS = ["source", "service_date", "route_id", "direction_id", "stop_id", "scheduled",
           "trip_id", "stop_sequence"]
STAGING_DDL = """
CREATE TEMP TABLE IF NOT EXISTS api_staging (
    source text, service_date date, route_id text, direction_id text, stop_id text,
    scheduled timestamp, trip_id text, stop_sequence smallint
)
"""
_cols = ", ".join(COLUMNS)
INSERT = f"""
INSERT INTO stop_events ({_cols})
SELECT {_cols} FROM api_staging
ON CONFLICT ON CONSTRAINT stop_events_key DO NOTHING
"""


class Api:
    def __init__(self, api_key=None):
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/vnd.api+json"
        if api_key:
            self.session.headers["x-api-key"] = api_key
        self.interval = KEY_INTERVAL_S if api_key else NO_KEY_INTERVAL_S
        self._last = 0.0

    def get(self, path, params=None, attempts=5):
        """GET one endpoint, following pagination, pacing requests and retrying 429/5xx."""
        url, records = f"{API_BASE}{path}", []
        while url:
            for attempt in range(1, attempts + 1):
                time.sleep(max(0.0, self._last + self.interval - time.monotonic()))
                self._last = time.monotonic()
                try:
                    resp = self.session.get(url, params=params, timeout=(10, 60))
                except (requests.ConnectionError, requests.Timeout):
                    if attempt == attempts:
                        raise
                    time.sleep(2 ** attempt)
                    continue
                if resp.status_code == 429 or resp.status_code >= 500:
                    if attempt == attempts:
                        resp.raise_for_status()
                    time.sleep(max(2 ** attempt, int(resp.headers.get("retry-after", 0) or 0)))
                    continue
                resp.raise_for_status()
                break
            payload = resp.json()
            records.extend(payload.get("data", []))
            url, params = payload.get("links", {}).get("next"), None
        return records


def api_route_ids(file_routes, api_routes):
    """Map file-style route ids to API route ids.

    file_routes: route_id values from the historical table ('01', '66', 'SL1').
    api_routes: {api_id: short_name} from /routes?filter[type]=3.
    Returns ({file_id: api_id}, [file ids the API doesn't know]).
    """
    by_short_name = {short: api_id for api_id, short in api_routes.items() if short}
    mapping, unknown = {}, []
    for route in sorted(file_routes):
        stripped = route.lstrip("0") or "0"
        if stripped in api_routes:
            mapping[route] = stripped
        elif route in by_short_name:
            mapping[route] = by_short_name[route]
        elif stripped in by_short_name:
            mapping[route] = by_short_name[stripped]
        else:
            unknown.append(route)
    return mapping, unknown


def map_record(item, file_route, service_date):
    """One /schedules record -> a stop_events row (file conventions), or None if unusable."""
    attrs, rels = item.get("attributes", {}), item.get("relationships", {})
    direction = DIRECTIONS.get(attrs.get("direction_id"))
    stop_id = (rels.get("stop", {}).get("data") or {}).get("id")
    when = attrs.get("departure_time") or attrs.get("arrival_time")  # last stop has arrival only
    if direction is None or not stop_id or not when:
        return None
    # Drop the UTC offset so the timestamp is Boston wall clock, like the files.
    scheduled = datetime.fromisoformat(when).replace(tzinfo=None)
    return {
        "source": "api",
        "service_date": service_date,
        "route_id": file_route,
        "direction_id": direction,
        "stop_id": stop_id,
        "scheduled": scheduled.isoformat(sep=" "),
        "trip_id": (rels.get("trip", {}).get("data") or {}).get("id"),
        "stop_sequence": attrs.get("stop_sequence"),
    }


def insert_rows(conn, rows):
    buf = io.StringIO()
    for row in rows:
        buf.write(",".join("" if row[c] is None else str(row[c]) for c in COLUMNS) + "\n")
    with conn.cursor() as cur:
        cur.execute("TRUNCATE api_staging")
        with cur.copy(f"COPY api_staging ({_cols}) FROM STDIN WITH (FORMAT csv)") as copy:
            copy.write(buf.getvalue())
        cur.execute(INSERT)
        inserted = cur.rowcount
    conn.commit()
    return inserted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", help="service date YYYY-MM-DD (default: today in Boston)")
    parser.add_argument("--routes", nargs="*", help="file-style route ids to pull (default: all in the table)")
    args = parser.parse_args(argv)
    service_date = args.date or datetime.now(BOSTON).date().isoformat()
    api = Api(os.environ.get("MBTA_API_KEY"))
    started = time.monotonic()

    with psycopg.connect(database_url()) as conn:
        conn.execute(STAGING_DDL)
        timepoints = set(conn.execute(
            "SELECT DISTINCT route_id, direction_id, stop_id FROM stop_events WHERE source = 'historical'"
        ).fetchall())
        if not timepoints:
            raise SystemExit("no historical rows in stop_events; run src.load_historical first")
        file_routes = {r for r, _, _ in timepoints}
        if args.routes:
            file_routes &= set(args.routes)

        api_routes = {r["id"]: r["attributes"].get("short_name")
                      for r in api.get("/routes", {"filter[type]": "3"})}
        mapping, unknown = api_route_ids(file_routes, api_routes)
        if unknown:
            print(f"routes the API doesn't know (skipped): {', '.join(unknown)}", flush=True)

        fetched = kept = inserted = 0
        empty = []
        for i, (file_route, api_id) in enumerate(sorted(mapping.items()), 1):
            records = api.get("/schedules", {"filter[route]": api_id, "filter[date]": service_date})
            if not records:
                empty.append(file_route)
                continue
            rows = [r for r in (map_record(rec, file_route, service_date) for rec in records)
                    if r and (r["route_id"], r["direction_id"], r["stop_id"]) in timepoints]
            fetched += len(records)
            kept += len(rows)
            inserted += insert_rows(conn, rows) if rows else 0
            print(f"[{i}/{len(mapping)}] route {file_route} (api {api_id}): "
                  f"{len(records):,} records, {len(rows):,} timepoint rows", flush=True)

    summary = {
        "run_at": datetime.now(BOSTON).isoformat(timespec="seconds"),
        "service_date": service_date,
        "routes_requested": len(mapping),
        "routes_with_service": len(mapping) - len(empty),
        "routes_unknown_to_api": unknown,
        "api_records_fetched": fetched,
        "timepoint_rows_kept": kept,
        "rows_inserted": inserted,
        "seconds": round(time.monotonic() - started, 1),
    }
    print(json.dumps(summary, indent=2))
    LOG_FILE.parent.mkdir(exist_ok=True)
    with LOG_FILE.open("a") as log:
        log.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
