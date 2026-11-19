"""Inputs for the latenet neural network, built from the frozen splits only."""

"""Inputs for the latenet neural network, built from the frozen splits only."""

import numpy as np
import pandas as pd
from src.db import ROOT

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

def build_scaling(train):
    """Numbers needed to scale timepoint order, from the train split only."""
    order = train["time_point_order"].astype("float64")
    return {"time_point_order_fill": float(order.median()),
            "time_point_order_max": float(order.max())}


def numeric_inputs(df, scaling):
    """Direction, weekend and standard type as 0/1 flags, plus scaled timepoint order."""
    order = df["time_point_order"].astype("float64").fillna(scaling["time_point_order_fill"])
    columns = [
        df["direction_id"].astype(str) == "Inbound",
        df["day_of_week"].isin([5, 6]),
        df["standard_type"].astype(str) == "Headway",
        order / scaling["time_point_order_max"],
    ]
    return np.column_stack(columns).astype("float32")

def make_loader(df, vocab, scaling, batch_size, shuffle):
    """Tensors for one split: category indexes, numeric inputs and labels, in batches."""
    idx = encode(df, vocab)
    cat = torch.from_numpy(np.column_stack([idx[c] for c in CATEGORICAL]))   # int64, one column per category
    num = torch.from_numpy(numeric_inputs(df, scaling))                       # float32, 4 columns
    y = torch.from_numpy(df[LABEL].to_numpy().astype("float32"))             # 0.0 / 1.0
    return DataLoader(TensorDataset(cat, num, y), batch_size=batch_size, shuffle=shuffle)