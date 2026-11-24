"""Where is the model wrong most often? Error breakdowns on its validation predictions.

    python -m src.failure_cases final --threshold 0.40

Reads data/preds/<run>_val.parquet (written by src.train), applies the same rolling 7-day
recalibration as serving, and breaks down two kinds of error at the threshold:
  missed late:  a late bus shown as "Usually on time" (the spec's expensive mistake)
  false alarm:  an on-time bus that got a warning
by hour, by day type, by day and by route. Validation only; the test split is not read.
"""

import argparse

import numpy as np
import pandas as pd

from src.calibrate import apply_shifts, rolling_shifts
from src.features import add_day_type
from src.train import PRED_DIR

MIN_ROUTE_ROWS = 5000


def breakdown(df, key):
    g = df.groupby(key, observed=True)
    out = pd.DataFrame({
        "rows": g.size(),
        "actual_late": g["late"].mean(),
        "predicted": g["p"].mean(),
        "error_rate": g["wrong"].mean(),
        "missed_late": g.apply(lambda x: (~x["warn"] & x["late"]).sum() / max(x["late"].sum(), 1)),
        "false_alarm": g.apply(lambda x: (x["warn"] & ~x["late"]).sum() / max((~x["late"]).sum(), 1)),
    })
    out["gap"] = out["predicted"] - out["actual_late"]
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_name")
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--window", type=int, default=7)
    args = parser.parse_args(argv)

    df = pd.read_parquet(PRED_DIR / f"{args.run_name}_val.parquet")
    df["p"] = apply_shifts(df["service_date"], df["proba"],
                           rolling_shifts(df["service_date"], df["proba"], df["late"], args.window))
    df["late"] = df["late"].astype(bool)
    df["warn"] = df["p"] >= args.threshold
    df["wrong"] = df["warn"] != df["late"]
    df = add_day_type(df)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")

    overall = breakdown(df.assign(all="all"), "all")
    print(f"{args.run_name} on validation at {args.threshold} (recalibrated):")
    print(overall.to_string())
    print("\nBy hour (worst calibration gaps are the hours to watch):")
    print(breakdown(df, "hour").to_string())
    print("\nBy day type:")
    print(breakdown(df, "day_type").to_string())
    print("\nBy day:")
    print(breakdown(df, "service_date").to_string())
    routes = breakdown(df, "route_id")
    routes = routes[routes["rows"] >= MIN_ROUTE_ROWS]
    print(f"\nRoutes with the most missed late buses ({len(routes)} routes with >= {MIN_ROUTE_ROWS:,} rows):")
    print(routes.sort_values("missed_late", ascending=False).head(5).to_string())
    print("\nRoutes with the most false alarms:")
    print(routes.sort_values("false_alarm", ascending=False).head(5).to_string())
    r = np.corrcoef(routes["predicted"], routes["actual_late"])[0, 1]
    print(f"\nRoute-level Pearson r, predicted vs actual late rate: {r:.3f}")


if __name__ == "__main__":
    main()
