"""Train a model on the train split, score it on validation, log the run to MLflow.

    python -m src.train configs/logreg_history.yaml

Config keys:
  features          categorical columns (one-hot for logreg)
  history           optional {windows, smoothing, counts}: adds recent late-rate features
                    from src/features.py, each row using only earlier service days
  numeric_features  optional extra numeric columns
  train.burn_in_days  optional: drop the first N days of train, whose history is thin
  model.type        majority | logreg | hgb | history_rate (params.column is the probability)

The test split is never read here. Validation predictions are saved to
data/preds/<run_name>_val.parquet for calibration and threshold work, and the run table is
exported to mlflow_runs.csv.
"""

import argparse
import subprocess
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, log_loss,
                             precision_score, recall_score, roc_auc_score)

from src.db import ROOT
from src.models import TRUSTED_TYPES, build_model
from src.extra import extra_md5s, keep_train_rows, load_extra
from src.features import add_day_type, add_history_features, history_feature_names

SPLIT_DIR = ROOT / "data" / "splits"
PRED_DIR = ROOT / "data" / "preds"
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
ARTIFACT_ROOT = (ROOT / "mlruns").as_uri()
RUNS_CSV = ROOT / "mlflow_runs.csv"
CATEGORICAL = ["route_id", "direction_id", "time_point_id", "point_type", "standard_type", "day_type"]


def load_frames(cfg):
    """Train and validation with day_type and (optionally) history features. Test is not read."""
    train = pd.read_parquet(SPLIT_DIR / "train.parquet")
    val = pd.read_parquet(SPLIT_DIR / "val.parquet")
    train_start = train["service_date"].min()
    extra = load_extra(cfg)  # optional earlier months (src/extra.py)
    both = add_day_type(pd.concat([train, val] if extra is None else [extra, train, val], ignore_index=True))
    # One shared category set for train and val (concat of differing categories gives object).
    for c in CATEGORICAL:
        both[c] = both[c].astype(str).astype("category")
    hist = cfg.get("history")
    if hist:
        both = add_history_features(both, both, windows=tuple(hist["windows"]), smoothing=hist["smoothing"])
    is_val = both["service_date"] >= val["service_date"].min()
    train, val = both[~is_val], both[is_val]
    train = keep_train_rows(train, cfg, train_start)
    burn_in = cfg.get("train", {}).get("burn_in_days", 0)
    if burn_in:
        train = train[train["service_date"] >= train["service_date"].min() + pd.Timedelta(days=burn_in)]
    return train, val


def feature_lists(cfg):
    categorical = list(cfg["features"])
    numeric = list(cfg.get("numeric_features", []))
    hist = cfg.get("history")
    if hist:
        numeric += history_feature_names(tuple(hist["windows"]), counts=hist.get("counts", True))
    return categorical, numeric


def model_inputs(df, categorical, numeric):
    X = df[categorical + numeric].copy()
    X[numeric] = X[numeric].astype("float32")  # nullable ints (headway) become NaN
    return X


def score(y, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "val_accuracy": accuracy_score(y, pred),
        "val_auc_roc": roc_auc_score(y, proba),
        "val_pr_auc": average_precision_score(y, proba),
        "val_precision": precision_score(y, pred, zero_division=0),
        "val_recall": recall_score(y, pred),
        "val_flagged_share": float(pred.mean()),
        "val_mean_pred": float(proba.mean()),
        "val_brier": brier_score_loss(y, proba),
        "val_log_loss": log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6)),
    }


def dvc_md5(name):
    pointer = SPLIT_DIR / f"{name}.parquet.dvc"
    return yaml.safe_load(pointer.read_text())["outs"][0]["md5"] if pointer.exists() else "untracked"


def git_commit():
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "configs"],
                           capture_output=True, text=True, cwd=ROOT).stdout.strip()
    return sha + ("-dirty" if dirty else "")


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = ",".join(map(str, v)) if isinstance(v, list) else v
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    label = cfg["label"]
    categorical, numeric = feature_lists(cfg)

    train, val = load_frames(cfg)
    X_train, y_train = model_inputs(train, categorical, numeric), train[label].to_numpy()
    X_val, y_val = model_inputs(val, categorical, numeric), val[label].to_numpy()

    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(cfg["experiment"]) is None:
        mlflow.create_experiment(cfg["experiment"], artifact_location=ARTIFACT_ROOT)
    mlflow.set_experiment(cfg["experiment"])

    with mlflow.start_run(run_name=cfg["run_name"]) as run:
        mlflow.log_params(flatten({k: v for k, v in cfg.items() if k not in ("experiment", "run_name")}))
        mlflow.log_params({
            "git_commit": git_commit(),
            "dvc_md5_train": dvc_md5("train"),
            "dvc_md5_val": dvc_md5("val"),
            "dvc_md5_test": dvc_md5("test"),
            **extra_md5s(cfg, dvc_md5),
            "train_first_day": str(train["service_date"].min().date()),
            "rows_train": len(y_train),
            "rows_val": len(y_val),
            "n_numeric_features": len(numeric),
        })
        mlflow.log_artifact(str(args.config), artifact_path="config")

        model = build_model(cfg, categorical, numeric)
        started = time.perf_counter()
        model.fit(X_train, y_train)
        fit_s = time.perf_counter() - started

        proba = model.predict_proba(X_val)[:, 1]
        metrics = {**score(y_val, proba), "val_late_rate": float(y_val.mean()),
                   "train_late_rate": float(y_train.mean()), "fit_seconds": fit_s}
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model", skops_trusted_types=TRUSTED_TYPES)

        PRED_DIR.mkdir(parents=True, exist_ok=True)
        preds = val[["service_date", "route_id", "direction_id", "stop_id", "hour", label]].copy()
        preds["proba"] = proba.astype("float32")
        pred_path = PRED_DIR / f"{cfg['run_name']}_val.parquet"
        preds.to_parquet(pred_path, index=False)
        mlflow.log_artifact(str(pred_path), artifact_path="predictions")
        mlflow.set_tag("val_predictions", pred_path.relative_to(ROOT).as_posix())
        print(f"{cfg['run_name']} ({run.info.run_id}): " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    export_runs(cfg["experiment"])


def export_runs(experiment):
    """Write every run to mlflow_runs.csv, best validation AUC first (best epoch for the nets)."""
    runs = mlflow.search_runs(experiment_names=[experiment])
    runs.insert(1, "run_name", runs["tags.mlflow.runName"])
    best = runs["metrics.best_val_auc_roc"] if "metrics.best_val_auc_roc" in runs else runs["metrics.val_auc_roc"]
    runs.insert(2, "val_auc_for_ranking", best.fillna(runs["metrics.val_auc_roc"]))
    runs = runs.sort_values("val_auc_for_ranking", ascending=False)
    runs.to_csv(RUNS_CSV, index=False)
    print(f"wrote {len(runs)} runs to {RUNS_CSV.name}")


if __name__ == "__main__":
    main()
