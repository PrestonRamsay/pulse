"""Rolling recalibration and threshold tables for a trained run's validation predictions.

    python -m src.calibrate gbt_history

A model trained on summer keeps its summer level; a one-off calibration fitted on 1-7 Sep did
not carry into the school weeks. Rolling recalibration instead shifts each day's predictions,
on the log-odds scale, by the amount that would have made the previous `window` days'
predictions average out to those days' actual late rate. Like the history features it uses
only earlier service days, and only out-of-sample predictions (never the model's own training
rows), so the first validation day is left unshifted.

Reads data/preds/<run>_val.parquet (written by src.train), logs cal_* metrics and a threshold
table to that MLflow run.
"""

import argparse

import mlflow
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.db import ROOT
from src.train import PRED_DIR, TRACKING_URI

THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
CHECK_FROM = "2026-09-08"  # second validation week: schools back, the hardest drift


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def level_shift(proba, actual_rate):
    """Log-odds shift b with mean(sigmoid(logit(proba) + b)) == actual_rate."""
    z = logit(np.asarray(proba, dtype=float))
    rate = float(np.clip(actual_rate, 1e-4, 1 - 1e-4))
    return brentq(lambda b: sigmoid(z + b).mean() - rate, -10, 10)


def rolling_shifts(dates, proba, y, window=7):
    """{service_date: shift} from the `window` days before each date (0.0 with no history)."""
    df = pd.DataFrame({"d": pd.to_datetime(dates), "p": proba, "y": y})
    shifts = {}
    for day in sorted(df["d"].unique()):
        past = df[(df["d"] < day) & (df["d"] >= day - pd.Timedelta(days=window))]
        shifts[day] = level_shift(past["p"], past["y"].mean()) if len(past) else 0.0
    return shifts


def apply_shifts(dates, proba, shifts):
    b = pd.to_datetime(pd.Series(dates)).map(shifts).fillna(0.0).to_numpy()
    return sigmoid(logit(np.asarray(proba, dtype=float)) + b)


def ece(y, p, bins=10):
    """Expected calibration error: |actual - predicted| per probability decile, row-weighted."""
    y, p = np.asarray(y), np.asarray(p)
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[b == i].mean() - p[b == i].mean()) * (b == i).mean()
                     for i in range(bins) if (b == i).any()))


def calibration_metrics(y, p):
    return {"mean_pred": float(np.mean(p)), "late_rate": float(np.mean(y)), "ece": ece(y, p),
            "brier": brier_score_loss(y, p), "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
            "auc_roc": roc_auc_score(y, p)}


def threshold_table(y, p, thresholds=THRESHOLDS):
    """Per threshold: share warned, recall, precision, and how often an unwarned trip ran late."""
    y, p = np.asarray(y), np.asarray(p)
    rows = []
    for t in thresholds:
        warn = p >= t
        rows.append({"threshold": t, "warned_share": warn.mean(),
                     "recall": y[warn].sum() / y.sum(),
                     "precision": y[warn].mean() if warn.any() else np.nan,
                     "unwarned_late_rate": y[~warn].mean() if (~warn).any() else np.nan,
                     "on_time_flagged": warn[y == 0].mean() if (y == 0).any() else np.nan,
                     "accuracy": (warn == (y == 1)).mean()})
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_name")
    parser.add_argument("--window", type=int, default=7)
    args = parser.parse_args(argv)

    preds = pd.read_parquet(PRED_DIR / f"{args.run_name}_val.parquet")
    shifts = rolling_shifts(preds["service_date"], preds["proba"], preds["late"], args.window)
    preds["proba_cal"] = apply_shifts(preds["service_date"], preds["proba"], shifts)
    check = preds[preds["service_date"] >= CHECK_FROM]

    metrics = {}
    for scope, part in [("val", preds), ("wk2", check)]:
        for kind, col in [("raw", "proba"), ("cal", "proba_cal")]:
            for k, v in calibration_metrics(part["late"], part[col]).items():
                metrics[f"{scope}_{kind}_{k}"] = v
    table = threshold_table(check["late"], check["proba_cal"])
    table_all = threshold_table(preds["late"], preds["proba_cal"])  # all of validation (1-15 Sep)

    print(f"{args.run_name}: rolling {args.window}-day recalibration")
    for scope in ("val", "wk2"):
        for kind in ("raw", "cal"):
            print(f"  {scope} {kind}: " + ", ".join(
                f"{k}={metrics[f'{scope}_{kind}_{k}']:.4f}" for k in ("mean_pred", "late_rate", "ece", "brier", "auc_roc")))
    fmt = lambda v: f"{v:.3f}"
    print("  thresholds on all validation days, recalibrated:")
    print(table_all.to_string(index=False, float_format=fmt))
    print(f"  thresholds on {CHECK_FROM} onward, recalibrated:")
    print(table.to_string(index=False, float_format=fmt))

    mlflow.set_tracking_uri(TRACKING_URI)
    runs = mlflow.search_runs(experiment_names=["pulse-late-3min"],
                              filter_string=f"tags.mlflow.runName = '{args.run_name}'",
                              order_by=["start_time DESC"], max_results=1)
    with mlflow.start_run(run_id=runs["run_id"].iloc[0]):
        mlflow.log_param("cal_window_days", args.window)
        mlflow.log_metrics({f"cal_{k}": v for k, v in metrics.items()})
        out = PRED_DIR / f"{args.run_name}_thresholds.csv"
        table.to_csv(out, index=False)
        mlflow.log_artifact(str(out), artifact_path="calibration")
        out_all = PRED_DIR / f"{args.run_name}_thresholds_val_all.csv"
        table_all.to_csv(out_all, index=False)
        mlflow.log_artifact(str(out_all), artifact_path="calibration")
    print(f"logged to run {runs['run_id'].iloc[0]} ({out.relative_to(ROOT).as_posix()})")


if __name__ == "__main__":
    main()
