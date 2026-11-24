"""Do LAMP's actual times agree with the monthly files? Compare them on days both cover.

    python -m src.lamp_check                                  # 1-15 Sep 2026 (validation weeks)
    python -m src.lamp_check --start 2026-09-01 --end 2026-09-15

Reads LAMP rows for the range from the big LAMP file (only the needed row groups and
columns, ~25 MB for two weeks), converts them with src/lamp.py, and matches them to the
monthly-file rows in stop_events on (service date, route, direction, stop, scheduled time).
No model is involved. The default range is the validation weeks, to stay off the test split.
"""

import argparse

import pandas as pd
import psycopg

from src.db import database_url
from src.lamp import KEY, is_late, read_all_range, to_stop_events

QUERY = """
SELECT service_date, route_id, direction_id, stop_id, scheduled, actual, late, point_type
FROM stop_events
WHERE source = 'historical' AND actual IS NOT NULL AND service_date BETWEEN %s AND %s
"""


def compare(monthly, lamp):
    """Match on the key; returns (summary dict, matched rows)."""
    j = monthly.merge(lamp[KEY + ["actual"]], on=KEY, how="inner", suffixes=("", "_lamp"))
    gap = (j["actual_lamp"] - j["actual"]).dt.total_seconds().abs()
    j["late_lamp"] = is_late(j.rename(columns={"actual": "actual_month", "actual_lamp": "actual"}))
    summary = {
        "monthly_rows": len(monthly),
        "lamp_rows": len(lamp),
        "matched": len(j),
        "matched_share": len(j) / len(monthly),
        "actual_identical": float((gap == 0).mean()),
        "actual_within_60s": float((gap <= 60).mean()),
        "label_agreement": float((j["late_lamp"] == j["late"].astype(bool)).mean()),
        "late_rate_monthly": float(j["late"].astype(bool).mean()),
        "late_rate_lamp": float(j["late_lamp"].mean()),
    }
    return summary, j


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=pd.Timestamp, default=pd.Timestamp("2026-09-01"))
    parser.add_argument("--end", type=pd.Timestamp, default=pd.Timestamp("2026-09-15"))
    args = parser.parse_args(argv)
    start, end = args.start.date(), args.end.date()

    with psycopg.connect(database_url()) as conn:
        monthly = pd.read_sql(QUERY, conn, params=(start, end))
    monthly["service_date"] = pd.to_datetime(monthly["service_date"]).dt.date
    lamp, counts = to_stop_events(read_all_range(start, end))

    summary, j = compare(monthly, lamp)
    print(f"LAMP vs monthly files, {start} to {end}")
    print(f"  LAMP rows read {counts['lamp_rows']:,}; with an actual time {counts['with_actual']:,}; "
          f"kept {counts['rows']:,}")
    for k, v in summary.items():
        print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v:,}")
    by_type = j.groupby("point_type").apply(lambda g: (g["late_lamp"] == g["late"].astype(bool)).mean())
    print("  label agreement by point type: " + ", ".join(f"{k} {v:.4f}" for k, v in by_type.items()))
    unmatched = monthly.merge(lamp[KEY], on=KEY, how="left", indicator=True)
    top = unmatched[unmatched["_merge"] == "left_only"]["route_id"].value_counts().head(5)
    print("  unmatched monthly rows, top routes: " + ", ".join(f"{r} {n:,}" for r, n in top.items()))


if __name__ == "__main__":
    main()
