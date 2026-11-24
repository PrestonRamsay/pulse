"""Export extra labelled months (before train) as a frozen file next to the splits.

    python -m src.make_extra --start 2026-04-01 --end 2026-06-30 --name apr_jun
    dvc add data/splits/apr_jun.parquet

Same query, columns and types as src/make_splits.py, restricted to a date range that must
end before the train split starts. Configs use it through `extra_data` (src/extra.py):
either as history only (it fills the late-rate windows, train rows unchanged) or as extra
training rows. Validation and test are never touched.
"""

import argparse
import tempfile
from pathlib import Path

import pandas as pd
import psycopg

from src.db import database_url
from src.make_splits import DTYPES, QUERY, SPLIT_DIR, add_calendar_features


def fetch_range(conn, start, end):
    query = f"{QUERY.strip()} AND service_date BETWEEN '{start}' AND '{end}'"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rows.csv"
        with conn.cursor() as cur, path.open("wb") as f:
            with cur.copy(f"COPY ({query}) TO STDOUT WITH (FORMAT csv, HEADER)") as copy:
                for block in copy:
                    f.write(block)
        df = pd.read_csv(path, dtype=DTYPES, parse_dates=["service_date", "scheduled"],
                         true_values=["t"], false_values=["f"])
    df["late"] = df["late"].astype("int8")
    return df


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", required=True, type=pd.Timestamp)
    parser.add_argument("--end", required=True, type=pd.Timestamp)
    parser.add_argument("--name", required=True)
    args = parser.parse_args(argv)

    train_start = pd.read_parquet(SPLIT_DIR / "train.parquet", columns=["service_date"])["service_date"].min()
    if args.end >= train_start:
        raise SystemExit(f"--end must be before the train split starts ({train_start:%Y-%m-%d}).")

    with psycopg.connect(database_url()) as conn:
        df = add_calendar_features(fetch_range(conn, f"{args.start:%Y-%m-%d}", f"{args.end:%Y-%m-%d}"))
    out = SPLIT_DIR / f"{args.name}.parquet"
    df.sort_values(["scheduled", "route_id", "stop_id"]).to_parquet(out, index=False)
    print(f"{args.name}: {len(df):,} rows, {df['service_date'].min():%Y-%m-%d} to "
          f"{df['service_date'].max():%Y-%m-%d}, late rate {df['late'].mean():.4f} -> {out}")


if __name__ == "__main__":
    main()
