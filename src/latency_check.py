"""Time a single prediction from the winning model on CPU (spec bar: under 200 ms).

    python -m src.latency_check
    python -m src.latency_check --model models/pulse-latenet --repeats 500

Loads the DVC-tracked model directory (models/pulse-latenet, an MLflow model exported from
pulse-latenet version 1), builds the inputs for one real departure (the first row of the
last validation day, with its late rates computed from earlier days only), then times
predict_proba on that single row.

What is timed is the serving call: one-row DataFrame in, probability out, plus the rolling
recalibration shift and the threshold, which are one addition and one comparison. The late
rates themselves are computed once a day in a batch job and looked up at serving time, so
building them is timed separately and not counted against the bar.
"""

import argparse
import platform
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

from src.db import ROOT
from src.features import add_day_type, add_history_features
from src.train import CATEGORICAL, SPLIT_DIR, feature_lists, model_inputs

BAR_MS = 200.0


def example_row(cfg):
    """Inputs for one validation departure, history from earlier service days only."""
    train = pd.read_parquet(SPLIT_DIR / "train.parquet")
    val = pd.read_parquet(SPLIT_DIR / "val.parquet")
    both = add_day_type(pd.concat([train, val], ignore_index=True))
    for c in CATEGORICAL:
        both[c] = both[c].astype(str).astype("category")
    target = both[both["service_date"] == both["service_date"].max()].head(1)
    h = cfg["history"]
    started = time.perf_counter()
    row = add_history_features(both, target, windows=tuple(h["windows"]), smoothing=h["smoothing"])
    history_s = time.perf_counter() - started
    categorical, numeric = feature_lists(cfg)
    return model_inputs(row, categorical, numeric), history_s


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "pulse-latenet")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "final.yaml")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.40)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    model = mlflow.sklearn.load_model(str(args.model))
    load_s = time.perf_counter() - started
    X, history_s = example_row(yaml.safe_load(args.config.read_text()))
    shift = 0.0  # the day's recalibration shift (log-odds), looked up like the late rates

    def serve(x):
        p = model.predict_proba(x)[:, 1]
        z = np.log(p / (1 - p)) + shift
        p = 1 / (1 + np.exp(-z))
        return p, p >= args.threshold

    for _ in range(10):  # warm-up: first calls pay one-off import and cache costs
        serve(X)
    times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        p, warn = serve(X)
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)

    print(f"CPU: {platform.processor() or platform.machine()} ({platform.system()} {platform.release()})")
    print(f"model load: {load_s:.2f} s (once at startup); daily late-rate batch for one row: {history_s:.1f} s")
    print(f"single prediction, {args.repeats} calls: median {np.median(times):.2f} ms, "
          f"p95 {np.percentile(times, 95):.2f} ms, max {times.max():.2f} ms")
    print(f"example: probability {p[0]:.3f}, warn={bool(warn[0])}")
    verdict = "PASS" if np.percentile(times, 95) < BAR_MS else "FAIL"
    print(f"{verdict}: p95 {np.percentile(times, 95):.2f} ms against the {BAR_MS:.0f} ms bar")


if __name__ == "__main__":
    main()
