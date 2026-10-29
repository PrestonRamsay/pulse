"""Train a baseline on the train split, score it on validation, log the run to MLflow.

    python -m src.train_baseline configs/baseline_majority.yaml
    python -m src.train_baseline configs/baseline_logreg.yaml

The test split is never read here. After the runs, the run table is exported to
mlflow_runs.csv at the repo root.
"""

import argparse
import subprocess
import time
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.db import ROOT

SPLIT_DIR = ROOT / "data" / "splits"
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
ARTIFACT_ROOT = (ROOT / "mlruns").as_uri()
RUNS_CSV = ROOT / "mlflow_runs.csv"


def build_model(cfg):
    kind = cfg["model"]["type"]
    if kind == "majority":
        return DummyClassifier(strategy="most_frequent")
    if kind == "logreg":
        encode = ColumnTransformer([("onehot", OneHotEncoder(handle_unknown="ignore"), cfg["features"])])
        return Pipeline([("encode", encode), ("model", LogisticRegression(**cfg["model"]["params"]))])
    raise ValueError(f"unknown model type: {kind}")


def load_split(name, features, label):
    df = pd.read_parquet(SPLIT_DIR / f"{name}.parquet", columns=[*features, label])
    # One-hot works on strings; hour and day of week are categories here, not quantities.
    return df[features].astype(str), df[label].to_numpy()


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
    features, label = cfg["features"], cfg["label"]

    X_train, y_train = load_split("train", features, label)
    X_val, y_val = load_split("val", features, label)

    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(cfg["experiment"]) is None:
        mlflow.create_experiment(cfg["experiment"], artifact_location=ARTIFACT_ROOT)
    mlflow.set_experiment(cfg["experiment"])

    with mlflow.start_run(run_name=cfg["run_name"]):
        mlflow.log_params(flatten({k: v for k, v in cfg.items() if k not in ("experiment", "run_name")}))
        mlflow.log_params({
            "git_commit": git_commit(),
            "dvc_md5_train": dvc_md5("train"),
            "dvc_md5_val": dvc_md5("val"),
            "dvc_md5_test": dvc_md5("test"),
            "rows_train": len(y_train),
            "rows_val": len(y_val),
        })
        mlflow.log_artifact(str(args.config), artifact_path="config")

        model = build_model(cfg)
        started = time.perf_counter()
        model.fit(X_train, y_train)
        fit_s = time.perf_counter() - started

        proba = model.predict_proba(X_val)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = {
            "val_accuracy": accuracy_score(y_val, pred),
            "val_auc_roc": roc_auc_score(y_val, proba),
            "val_pr_auc": average_precision_score(y_val, proba),
            "val_late_rate": float(y_val.mean()),
            "train_late_rate": float(y_train.mean()),
            "fit_seconds": fit_s,
        }
        mlflow.log_metrics(metrics)
        print(f"{cfg['run_name']}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    runs = mlflow.search_runs(experiment_names=[cfg["experiment"]])
    runs.to_csv(RUNS_CSV, index=False)
    print(f"wrote {len(runs)} runs to {RUNS_CSV.name}")


if __name__ == "__main__":
    main()
