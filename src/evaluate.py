"""The single final evaluation on the held-out test split, then register the model.

    python -m src.evaluate logreg_history_rates --threshold 0.40 --register

This is the only script that reads data/splits/test.parquet. It refuses to run twice for the
same run (the run gets test_* metrics the first time) unless --force is given, so the test
split is looked at once.

Steps:
1. Rebuild day type and history features over train + val + test. Each row still uses only
   earlier service days, so train and val features are unchanged by adding test.
2. Score the chosen run's model on test, then apply rolling recalibration: each test day is
   shifted by the previous 7 days' error (validation predictions seed the first test days).
3. Score the baselines on the same rows: majority class, the Task 2 schedule-only logistic
   regression (refit on train, deterministic), and the slot's 8-week late rate.
4. Metrics at the fixed threshold, confusion matrix, calibration by hour, top routes,
   alternative thresholds, and a day-level bootstrap of the AUC margin over each baseline.
5. With --register: register the model as pulse-latenet and, if it beats every baseline on
   test AUC with the bootstrap interval above zero, promote it to Production.
"""

import argparse
import json

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)

from src.calibrate import apply_shifts, ece, rolling_shifts, threshold_table
from src.db import ROOT
from src.features import add_day_type, add_history_features
from src.models import build_model
from src.train import CATEGORICAL, PRED_DIR, SPLIT_DIR, TRACKING_URI, dvc_md5, feature_lists, git_commit, model_inputs

MODEL_NAME = "pulse-latenet"
EXPERIMENT = "pulse-late-3min"
ALT_THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
MIN_ROUTE_ROWS = 2000  # routes smaller than this are too noisy for the top-5 check


def load_all(cfg):
    names = ("train", "val", "test")
    parts = [pd.read_parquet(SPLIT_DIR / f"{n}.parquet") for n in names]
    starts = [p["service_date"].min() for p in parts]
    df = add_day_type(pd.concat(parts, ignore_index=True))
    for c in CATEGORICAL:
        df[c] = df[c].astype(str).astype("category")
    hist = cfg.get("history")
    if hist:
        df = add_history_features(df, df, windows=tuple(hist["windows"]), smoothing=hist["smoothing"])
    d = df["service_date"]
    train = df[d < starts[1]]
    burn_in = cfg.get("train", {}).get("burn_in_days", 0)
    train = train[train["service_date"] >= starts[0] + pd.Timedelta(days=burn_in)]
    return train, df[(d >= starts[1]) & (d < starts[2])], df[d >= starts[2]]


def metrics_at(y, p, threshold):
    pred = p >= threshold
    return {"accuracy": accuracy_score(y, pred), "auc_roc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "warned_share": float(pred.mean()),
            "unwarned_late_rate": float(y[~pred].mean()), "mean_pred": float(p.mean()),
            "late_rate": float(y.mean()), "brier": brier_score_loss(y, p), "ece": ece(y, p)}


def bootstrap_auc_margin(dates, y, p_model, p_base, reps=200, seed=0):
    """AUC(model) - AUC(baseline) resampling whole service days; returns (2.5%, 97.5%)."""
    rng = np.random.default_rng(seed)
    days = np.unique(dates)
    idx_by_day = {d: np.flatnonzero(dates == d) for d in days}
    diffs = []
    for _ in range(reps):
        idx = np.concatenate([idx_by_day[d] for d in rng.choice(days, len(days), replace=True)])
        diffs.append(roc_auc_score(y[idx], p_model[idx]) - roc_auc_score(y[idx], p_base[idx]))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_name")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT], order_by=["start_time DESC"],
                              filter_string=f"tags.mlflow.runName = '{args.run_name}' and attributes.status = 'FINISHED'")
    run_id = runs["run_id"].iloc[0]
    if "metrics.test_auc_roc" in runs and pd.notna(runs["metrics.test_auc_roc"].iloc[0]) and not args.force:
        raise SystemExit(f"{args.run_name} already has test metrics: the test split is evaluated once.")
    cfg = yaml.safe_load((ROOT / "configs" / f"{args.run_name}.yaml").read_text())
    if cfg.get("extra_data"):
        raise SystemExit("extra_data runs aren't supported here (test windows would differ from training); "
                         "and test is not to be scored again without the reviewer's OK.")
    categorical, numeric = feature_lists(cfg)

    train, val, test = load_all(cfg)
    y = test["late"].to_numpy()
    dates = test["service_date"].to_numpy()

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    raw = model.predict_proba(model_inputs(test, categorical, numeric))[:, 1]

    val_preds = pd.read_parquet(PRED_DIR / f"{args.run_name}_val.parquet")
    seq = pd.DataFrame({"d": np.concatenate([val_preds["service_date"].to_numpy(), dates]),
                        "p": np.concatenate([val_preds["proba"].to_numpy(), raw]),
                        "y": np.concatenate([val_preds["late"].to_numpy(), y])})
    shifts = rolling_shifts(seq["d"], seq["p"], seq["y"], args.window)
    cal = apply_shifts(dates, raw, shifts)

    # Baselines on the same test rows.
    task2 = yaml.safe_load((ROOT / "configs" / "baseline_logreg.yaml").read_text())
    t2_feats = task2["features"]
    base_train = pd.read_parquet(SPLIT_DIR / "train.parquet", columns=[*t2_feats, "late"])
    t2 = build_model(task2, t2_feats, []).fit(base_train[t2_feats], base_train["late"])
    baselines = {
        "baseline_majority": np.zeros(len(y)),
        "baseline_logreg": t2.predict_proba(test[t2_feats])[:, 1],
        "history_slot_8wk": test["rate_slot_56d"].to_numpy(dtype=float),
    }

    results = {"model_raw": metrics_at(y, raw, args.threshold), "model": metrics_at(y, cal, args.threshold)}
    for name, p in baselines.items():
        results[name] = metrics_at(y, p, 0.5)
        if name != "baseline_majority":
            results[name]["auc_margin_ci"] = bootstrap_auc_margin(dates, y, cal, p)
    results["baseline_majority"]["auc_margin_ci"] = (results["model"]["auc_roc"] - 0.5,) * 2

    tn, fp, fn, tp = confusion_matrix(y, cal >= args.threshold).ravel()
    out = test[["service_date", "route_id", "hour", "late"]].copy()
    out["proba_raw"], out["proba"] = raw.astype("float32"), cal.astype("float32")
    by_hour = out.groupby("hour").agg(rows=("late", "size"), actual=("late", "mean"),
                                      predicted=("proba", "mean"), predicted_raw=("proba_raw", "mean"))
    routes = out.groupby("route_id", observed=True).agg(rows=("late", "size"), actual=("late", "mean"),
                                                        predicted=("proba", "mean"))
    routes = routes[routes["rows"] >= MIN_ROUTE_ROWS]
    top_pred = routes.sort_values("predicted", ascending=False).head(5)
    top_actual = routes.sort_values("actual", ascending=False).head(5)
    thresholds = threshold_table(y, cal, ALT_THRESHOLDS)
    daily = out.groupby("service_date").agg(actual=("late", "mean"), predicted=("proba", "mean"),
                                            predicted_raw=("proba_raw", "mean"))

    beats_all = all(results[b]["auc_margin_ci"][0] > 0 for b in baselines)

    # Save and print.
    tag = f"{args.run_name}_test"
    out.to_parquet(PRED_DIR / f"{tag}.parquet", index=False)
    tables = {"by_hour": by_hour, "routes_top_predicted": top_pred, "routes_top_actual": top_actual,
              "thresholds": thresholds, "daily": daily}
    for k, t in tables.items():
        t.to_csv(PRED_DIR / f"{tag}_{k}.csv")
    summary = {"run_id": run_id, "threshold": args.threshold, "window": args.window, "rows": int(len(y)),
               "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
               "results": results, "beats_all_baselines": beats_all,
               "git_commit": git_commit(), "dvc_md5_test": dvc_md5("test")}
    (PRED_DIR / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2, default=float))
    print(pd.DataFrame(results).T.drop(columns="auc_margin_ci", errors="ignore").round(4).to_string())
    for b in baselines:
        print(f"AUC margin over {b}: 95% day-bootstrap {results[b]['auc_margin_ci']}")
    for k, t in tables.items():
        print(f"\n{k}\n{t.round(4).to_string()}")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"test_{k}": v for k, v in results["model"].items() if not isinstance(v, tuple)})
        mlflow.log_metrics({f"test_raw_{k}": v for k, v in results["model_raw"].items()})
        mlflow.log_params({"test_threshold": args.threshold, "test_cal_window_days": args.window})
        for p in PRED_DIR.glob(f"{tag}_*"):
            if p.suffix in (".csv", ".json"):
                mlflow.log_artifact(str(p), artifact_path="test_evaluation")

    if args.register:
        mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
        for k, v in {"threshold": args.threshold, "recalibration": f"rolling {args.window}-day log-odds shift",
                     "test_auc_roc": round(results["model"]["auc_roc"], 4),
                     "test_accuracy": round(results["model"]["accuracy"], 4),
                     "source_run": args.run_name, "git_commit": summary["git_commit"]}.items():
            client.set_model_version_tag(MODEL_NAME, mv.version, k, str(v))
        client.update_model_version(MODEL_NAME, mv.version, description=(
            f"{args.run_name}: logistic regression on schedule fields + recent late rates. "
            f"Serve with rolling {args.window}-day recalibration; warn when probability >= {args.threshold}."))
        if beats_all:
            client.transition_model_version_stage(MODEL_NAME, mv.version, "Production", archive_existing_versions=True)
            client.set_registered_model_alias(MODEL_NAME, "production", mv.version)
            print(f"registered {MODEL_NAME} v{mv.version} and promoted to Production")
        else:
            print(f"registered {MODEL_NAME} v{mv.version}; NOT promoted: does not beat every baseline on test")


if __name__ == "__main__":
    main()
