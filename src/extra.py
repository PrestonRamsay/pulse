"""Extra labelled months that come before the train split (exported by src/make_extra.py).

A config opts in with:

    extra_data:
      files: [apr_jun]      # data/splits/<name>.parquet, DVC-tracked
      use: history          # history: fill the late-rate windows only, train rows unchanged
                            # train:   also train on these months

Validation and test rows are the same either way; only train rows (and, through the
windows, train-row features) change.
"""

import pandas as pd

from src.db import ROOT

SPLIT_DIR = ROOT / "data" / "splits"
USES = ("history", "train")


def extra_spec(cfg):
    spec = cfg.get("extra_data")
    if not spec:
        return None
    if spec.get("use") not in USES:
        raise ValueError(f"extra_data.use must be one of {USES}, got {spec.get('use')!r}")
    return spec


def load_extra(cfg, columns=None):
    """The extra months as one frame, or None when the config doesn't ask for any."""
    spec = extra_spec(cfg)
    if spec is None:
        return None
    return pd.concat([pd.read_parquet(SPLIT_DIR / f"{n}.parquet", columns=columns) for n in spec["files"]],
                     ignore_index=True)


def keep_train_rows(train, cfg, train_start):
    """With use: history, drop the extra months from the training rows (they only fed the windows)."""
    spec = extra_spec(cfg)
    if spec is not None and spec["use"] == "history":
        return train[train["service_date"] >= train_start]
    return train


def extra_md5s(cfg, dvc_md5):
    spec = extra_spec(cfg)
    return {} if spec is None else {f"dvc_md5_{n}": dvc_md5(n) for n in spec["files"]}
