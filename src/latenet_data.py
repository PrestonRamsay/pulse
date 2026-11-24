"""Inputs for the latenet neural network, built from the frozen splits only."""

import numpy as np
import pandas as pd
from src.db import ROOT
from src.extra import extra_spec, keep_train_rows, load_extra
from src.features import add_day_type, add_history_features, history_feature_names

import torch
from torch.utils.data import DataLoader, TensorDataset

SPLIT_DIR = ROOT / "data" / "splits"

CATEGORICAL = ["route_id", "stop_id", "hour", "day_of_week", "point_type"]
FLAGS = ["direction_id", "standard_type"]
NUMERIC = ["time_point_order"]
LABEL = "late"


def load_split(name):
    """Read one frozen split ('train' or 'val'), keeping only the model's columns."""
    columns = CATEGORICAL + FLAGS + NUMERIC + [LABEL]
    return pd.read_parquet(SPLIT_DIR / f"{name}.parquet", columns=columns)

def load_train_val(cfg):
    """Train and val inputs. With a `history` block in the config, also adds the recent
    late rates from src/features.py; each row uses only earlier service days.
    With `extra_data` (src/extra.py), earlier months either fill the history windows only or
    are also trained on. Returns (train, val, history_cols)."""
    h = cfg.get("history")
    if not h:
        extra = load_extra(cfg, columns=CATEGORICAL + FLAGS + NUMERIC + [LABEL])
        train = load_split("train")
        if extra is not None and extra_spec(cfg)["use"] == "train":
            train = pd.concat([extra, train], ignore_index=True)
        return train, load_split("val"), []
    train = pd.read_parquet(SPLIT_DIR / "train.parquet")
    val = pd.read_parquet(SPLIT_DIR / "val.parquet")
    train_start = train["service_date"].min()
    extra = load_extra(cfg)
    both = add_day_type(pd.concat([train, val] if extra is None else [extra, train, val], ignore_index=True))
    both = add_history_features(both, both, windows=tuple(h["windows"]), smoothing=h["smoothing"])
    history_cols = history_feature_names(tuple(h["windows"]), counts=False)
    is_val = both["service_date"] >= val["service_date"].min()
    train, val = both[~is_val], both[is_val]
    train = keep_train_rows(train, cfg, train_start)
    start = train["service_date"].min() + pd.Timedelta(days=h["burn_in_days"])
    train = train[train["service_date"] >= start]
    keep = CATEGORICAL + FLAGS + NUMERIC + history_cols + [LABEL]
    return train[keep], val[keep], history_cols

def load_eval_split(cfg, name):
    """One split ('val' or 'test') for scoring, with history rates if the model used them.
    History for each row comes only from earlier service days, the same as in training.
    Scoring 'val' reads train and val only; the test file is opened only for 'test'."""
    cols = CATEGORICAL + FLAGS + NUMERIC + [LABEL, "service_date"]
    h = cfg.get("history")
    if not h:
        return pd.read_parquet(SPLIT_DIR / f"{name}.parquet", columns=cols)
    order = ["train", "val", "test"]
    frames = [pd.read_parquet(SPLIT_DIR / f"{n}.parquet") for n in order[: order.index(name) + 1]]
    extra = load_extra(cfg)  # same windows as in training (src/extra.py)
    both = add_day_type(pd.concat(frames if extra is None else [extra, *frames], ignore_index=True))
    both = add_history_features(both, both, windows=tuple(h["windows"]), smoothing=h["smoothing"])
    history_cols = history_feature_names(tuple(h["windows"]), counts=False)
    target = both[both["service_date"] >= frames[-1]["service_date"].min()]
    return target[cols + history_cols]

def build_vocab(train):
    """One {value: index} table per categorical column, from the train split only.
    Index 0 is reserved for values never seen in training."""
    vocab = {}
    for col in CATEGORICAL:
        values = sorted(train[col].astype(str).unique())
        vocab[col] = {v: i + 1 for i, v in enumerate(values)}
    return vocab


def encode(df, vocab):
    """Replace each categorical value with its index; unseen values become 0."""
    out = {}
    for col in CATEGORICAL:
        out[col] = df[col].astype(str).map(vocab[col]).fillna(0).astype("int64").to_numpy()
    return out

def logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def build_scaling(train, history_cols=()):
    """Numbers needed to scale the numeric inputs, from the train split only."""
    order = train["time_point_order"].astype("float64")
    scaling = {"time_point_order_fill": float(order.median()),
               "time_point_order_max": float(order.max()),
               "history": {}}
    for col in history_cols:
        z = logit(train[col].to_numpy(dtype="float64"))
        scaling["history"][col] = [float(z.mean()), float(z.std())]
    return scaling


def numeric_inputs(df, scaling):
    """Flags, scaled timepoint order, and (if trained with history) standardized late rates."""
    order = df["time_point_order"].astype("float64").fillna(scaling["time_point_order_fill"])
    columns = [
        df["direction_id"].astype(str) == "Inbound",
        df["day_of_week"].isin([5, 6]),
        df["standard_type"].astype(str) == "Headway",
        order / scaling["time_point_order_max"],
    ]
    for col, (mean, std) in scaling.get("history", {}).items():
        columns.append((logit(df[col].to_numpy(dtype="float64")) - mean) / std)
    return np.column_stack(columns).astype("float32")

def make_loader(df, vocab, scaling, batch_size, shuffle):
    """Tensors for one split: category indexes, numeric inputs and labels, in batches."""
    idx = encode(df, vocab)
    cat = torch.from_numpy(np.column_stack([idx[c] for c in CATEGORICAL]))   # int64, one column per category
    num = torch.from_numpy(numeric_inputs(df, scaling))                       # float32, 4 columns + 1 per history rate
    y = torch.from_numpy(df[LABEL].to_numpy().astype("float32"))             # 0.0 / 1.0
    return DataLoader(TensorDataset(cat, num, y), batch_size=batch_size, shuffle=shuffle)