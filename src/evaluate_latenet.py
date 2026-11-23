"""Score a frozen split with a saved latenet model and print what REPORT §4 and §5 need.

    python -m src.evaluate_latenet val --threshold 0.45
    python -m src.evaluate_latenet val --model models/sweep/latenet_history.pt --threshold 0.45
"""

import argparse
from pathlib import Path

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

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=["val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model", type=Path, default=Path("models/latenet.pt"))
    args = parser.parse_args(argv)
    if args.split == "test":
        raise SystemExit("test is scored once, with --run-id (part 4). Use val for now.")

    df, saved = score_split(args.model, args.split)
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


if __name__ == "__main__":
    main()
