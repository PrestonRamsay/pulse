"""Score a frozen split with a saved latenet model and print what REPORT §4 and §5 need.

    python -m src.evaluate_latenet val --threshold 0.45
    python -m src.evaluate_latenet val --model models/sweep/latenet_history.pt --threshold 0.45
"""

import argparse
from pathlib import Path
import tempfile
from datetime import date

import mlflow

from src.train import TRACKING_URI

import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)

from src.latenet_data import load_eval_split, make_loader
from src.latenet_model import load_latenet
from src.train_latenet import predict


def score_split(model_path, split):
    """Rebuild the model's inputs for one split and add a 'proba' column."""
    model, saved = load_latenet(model_path)
    df = load_eval_split(saved["config"], split)
    loader = make_loader(df, saved["vocab"], saved["scaling"], batch_size=65536, shuffle=False)
    logits, _ = predict(model, loader)
    df = df.copy()
    df["proba"] = torch.sigmoid(logits).numpy()
    return df, saved


def headline(y, p, threshold):
    pred = p >= threshold
    return {"auc_roc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p),
            "accuracy_at_0.5": accuracy_score(y, p >= 0.5),
            f"accuracy_at_{threshold}": accuracy_score(y, pred),
            "precision": precision_score(y, pred),
            "recall": recall_score(y, pred)}


THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def by_hour(df):
    """Mean predicted probability against the actual late rate, per hour of day."""
    t = df.groupby("hour").agg(rows=("late", "size"), predicted=("proba", "mean"), actual=("late", "mean"))
    t["gap"] = t["predicted"] - t["actual"]
    return t


def route_check(df, min_rows=5000, top=5):
    """Top routes by predicted and by actual late rate, plus the Pearson r across routes."""
    r = df.groupby("route_id").agg(rows=("late", "size"), predicted=("proba", "mean"), actual=("late", "mean"))
    r = r[r["rows"] >= min_rows]
    pred_top = r.sort_values("predicted", ascending=False).head(top)
    act_top = r.sort_values("actual", ascending=False).head(top)
    side = pd.DataFrame({"top predicted": pred_top.index, "predicted": pred_top["predicted"].values,
                         "top actual": act_top.index, "actual": act_top["actual"].values})
    return side, r["predicted"].corr(r["actual"]), len(r)


def threshold_rows(y, p, thresholds=THRESHOLDS):
    """Per threshold: accuracy, precision, recall on late buses, share flagged,
    share of on-time buses flagged, and how often an unflagged bus still ran late."""
    rows = []
    for t in thresholds:
        warn = p >= t
        rows.append({"threshold": t, "accuracy": (warn == y).mean(),
                     "precision": y[warn].mean(), "recall_late": warn[y == 1].mean(),
                     "flagged_share": warn.mean(), "on_time_flagged": warn[y == 0].mean(),
                     "unwarned_late": y[~warn].mean()})
    return pd.DataFrame(rows)

def check_run(run_id, saved):
    """Refuse a second test scoring, and check models/latenet.pt really came from this run."""
    mlflow.set_tracking_uri(TRACKING_URI)
    run = mlflow.MlflowClient().get_run(run_id)
    if "test_auc_roc" in run.data.metrics:
        raise SystemExit(f"run {run_id} already has test metrics: the test split is scored once.")
    if abs(run.data.metrics["best_val_auc_roc"] - saved["best_val_auc_roc"]) > 1e-9:
        raise SystemExit("this model file doesn't match the run's best_val_auc_roc; wrong run id or file.")


def log_test(run_id, threshold, y, p, df, route_r):
    """Put the test metrics and tables on the winning run, so run history and report agree."""
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold).ravel()
    metrics = {f"test_{k}": float(v) for k, v in headline(y, p, threshold).items()}
    metrics.update({"test_tn": tn, "test_fp": fp, "test_fn": fn, "test_tp": tp,
                    "test_late_rate": y.mean(), "test_mean_pred": p.mean(),
                    "test_threshold": threshold, "test_route_pearson": route_r})
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.set_tag("test_scored_on", str(date.today()))
        with tempfile.TemporaryDirectory() as tmp:
            for name, table in [("by_hour", by_hour(df)), ("thresholds", threshold_rows(y, p))]:
                path = Path(tmp) / f"test_{name}.csv"
                table.to_csv(path)
                mlflow.log_artifact(str(path), artifact_path="test")
    print(f"\nlogged test metrics to run {run_id}")

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=["val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model", type=Path, default=Path("models/latenet.pt"))
    parser.add_argument("--run-id", help="the winning MLflow run; required for test")
    args = parser.parse_args(argv)
    if args.split == "test" and not args.run_id:
        raise SystemExit("test is scored once, and only with --run-id of the winning run.")
    if args.run_id and args.split != "test":
        raise SystemExit("--run-id is only for the single test run.")

    df, saved = score_split(args.model, args.split)
    if args.run_id:
        check_run(args.run_id, saved)
    y, p = df["late"].to_numpy(), df["proba"].to_numpy()
    print(f"{args.model} ({saved['config']['run_name']}) on {args.split}: {len(df):,} rows, "
          f"late rate {y.mean():.3f}, mean predicted {p.mean():.3f}")
    for k, v in headline(y, p, args.threshold).items():
        print(f"  {k}: {v:.4f}")

    tn, fp, fn, tp = confusion_matrix(y, p >= args.threshold).ravel()
    print(f"\nConfusion matrix at {args.threshold} (rows actual, columns predicted)")
    print(pd.DataFrame([[tn, fp], [fn, tp]], index=["actual on time", "actual late"],
                       columns=["pred on time", "pred late"]).to_string())

    print("\nCalibration by hour (mean predicted vs actual late rate)")
    print(by_hour(df).round(3).to_string())

    side, r, n = route_check(df)
    print(f"\nTop 5 routes, predicted vs actual ({n} routes with >= 5,000 rows); Pearson r = {r:.3f}")
    print(side.round(3).to_string(index=False))

    print("\nThreshold table")
    print(threshold_rows(y, p).round(3).to_string(index=False))

    if args.run_id:
        log_test(args.run_id, args.threshold, y, p, df, r)


if __name__ == "__main__":
    main()
