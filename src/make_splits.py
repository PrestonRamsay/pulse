"""Export labelled historical rows from stop_events and split them by time.

    python -m src.make_splits

Train = the two months before the latest loaded month, validation = days 1-15 of the
latest month, test = day 16 to month end. Writes data/splits/{train,val,test}.parquet.
Only serving-safe features (known before the trip runs) plus the label are exported.
"""

import argparse
import tempfile
from pathlib import Path

import pandas as pd
import psycopg

from src.db import ROOT, database_url

SPLIT_DIR = ROOT / "data" / "splits"

QUERY = """
SELECT service_date, route_id, direction_id, stop_id, time_point_id,
       scheduled, scheduled_headway, point_type, standard_type, time_point_order,
       late
FROM stop_events
WHERE source = 'historical' AND late IS NOT NULL
"""

DTYPES = {
    "route_id": "category", "direction_id": "category", "stop_id": "category",
    "time_point_id": "category", "point_type": "category", "standard_type": "category",
    "scheduled_headway": "Int32", "time_point_order": "Int16",
}


def add_calendar_features(df):
    df["day_of_week"] = df["scheduled"].dt.day_of_week.astype("int8")  # Monday = 0
    df["hour"] = df["scheduled"].dt.hour.astype("int8")
    df["minute_of_day"] = (df["scheduled"].dt.hour * 60 + df["scheduled"].dt.minute).astype("int16")
    return df


def split_by_date(df, train_months=2):
    """Cut on service_date: earlier months train, 1-15 of the latest month val, 16+ test."""
    latest = df["service_date"].max().to_period("M")
    month = df["service_date"].dt.to_period("M")
    day = df["service_date"].dt.day
    train = df[(month < latest) & (month >= latest - train_months)]
    val = df[(month == latest) & (day <= 15)]
    test = df[(month == latest) & (day >= 16)]
    return train, val, test


def check_order(train, val, test):
    """Every scheduled time in an earlier split precedes every one in the next."""
    assert train["scheduled"].max() < val["scheduled"].min(), "train overlaps validation"
    assert val["scheduled"].max() < test["scheduled"].min(), "validation overlaps test"


def fetch(conn):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rows.csv"
        with conn.cursor() as cur, path.open("wb") as f:
            with cur.copy(f"COPY ({QUERY}) TO STDOUT WITH (FORMAT csv, HEADER)") as copy:
                for block in copy:
                    f.write(block)
        df = pd.read_csv(path, dtype=DTYPES, parse_dates=["service_date", "scheduled"],
                         true_values=["t"], false_values=["f"])
    df["late"] = df["late"].astype("int8")
    return df


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-months", type=int, default=2)
    args = parser.parse_args(argv)

    with psycopg.connect(database_url()) as conn:
        df = add_calendar_features(fetch(conn))
    train, val, test = split_by_date(df, args.train_months)
    check_order(train, val, test)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.sort_values(["scheduled", "route_id", "stop_id"]).to_parquet(SPLIT_DIR / f"{name}.parquet", index=False)
        print(f"{name}: {len(part):,} rows, {part['service_date'].min():%Y-%m-%d} to "
              f"{part['service_date'].max():%Y-%m-%d}, scheduled {part['scheduled'].min()} to "
              f"{part['scheduled'].max()}, late rate {part['late'].mean():.4f}")


if __name__ == "__main__":
    main()
