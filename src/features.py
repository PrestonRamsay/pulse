"""Serving-safe features derived from the frozen splits.

Two kinds, both knowable before a trip runs:

* day_type: weekday / saturday / sunday from the service date's schedule pattern, not the
  weekday name. Holidays that ran a weekend schedule are mapped by hand (HOLIDAY_SERVICE);
  the published schedule says which pattern a holiday runs, so this is known in advance.
* Recent late rates: for each row, the late rate at several levels (network, route, stop,
  slot) over the last N service days *strictly before the row's own service date*. A row's
  own label, and every label from its own service day or later, never reaches its features.

Rates are shrunk toward the next-coarser level (slot -> stop -> route -> network -> 0.5) with
a pseudo-count, so a slot with three trips of history doesn't get a rate of 0 or 1.
"""

import numpy as np
import pandas as pd

# Service days that ran a different pattern than their weekday. Found from the daily row
# counts in the files (weekday ~85k rows, Saturday ~64k, Sunday ~54k) and matching holidays.
HOLIDAY_SERVICE = {
    "2026-07-03": "saturday",  # Independence Day observed (Fri): 67k rows
    "2026-07-04": "sunday",    # Independence Day (Sat): 54k rows
    "2026-09-07": "sunday",    # Labor Day (Mon): 56k rows
}

# Each level is shrunk toward the one before it.
LEVELS = {
    "network": [],
    "route": ["route_id"],
    "stop": ["route_id", "direction_id", "stop_id"],
    "slot": ["route_id", "direction_id", "stop_id", "hour", "day_type"],
}
DEFAULT_WINDOWS = (7, 14, 28, 56)


def add_day_type(df):
    dow = df["service_date"].dt.day_of_week
    day_type = np.where(dow == 5, "saturday", np.where(dow == 6, "sunday", "weekday"))
    day_type = pd.Series(day_type, index=df.index)
    holidays = df["service_date"].dt.strftime("%Y-%m-%d").map(HOLIDAY_SERVICE)
    df["day_type"] = holidays.fillna(day_type).astype("category")
    return df


def _key_codes(history, targets, keys):
    """One integer code per distinct key, shared by history and target rows."""
    if not keys:
        return np.zeros(len(history), dtype=np.int64), np.zeros(len(targets), dtype=np.int64), 1
    both = pd.concat([history[keys], targets[keys]], ignore_index=True)
    for k in keys:
        both[k] = both[k].astype(str)
    codes = both.groupby(keys, sort=False).ngroup().to_numpy()
    return codes[: len(history)], codes[len(history):], int(codes.max()) + 1


def window_counts(history, targets, keys, windows):
    """For each target row and window w: (trips, late trips) in history over the w service
    days before the target's service date (the target's own day excluded).

    Returns {w: (n, k)} with arrays aligned to targets.
    """
    origin = min(history["service_date"].min(), targets["service_date"].min())
    day_h = (history["service_date"] - origin).dt.days.to_numpy()
    day_t = (targets["service_date"] - origin).dt.days.to_numpy()
    n_days = int(max(day_h.max(), day_t.max())) + 1
    key_h, key_t, n_keys = _key_codes(history, targets, keys)

    # cum[key, i] = total over service days < i. History day d lands in column d + 1.
    width = n_days + 1
    flat = key_h * width + day_h + 1
    n = np.bincount(flat, minlength=n_keys * width).reshape(n_keys, width)
    k = np.bincount(flat, weights=history["late"].to_numpy(), minlength=n_keys * width).reshape(n_keys, width)
    n_cum, k_cum = n.cumsum(axis=1), k.cumsum(axis=1)

    out = {}
    for w in windows:
        start = np.maximum(day_t - w, 0)
        out[w] = (n_cum[key_t, day_t] - n_cum[key_t, start],
                  k_cum[key_t, day_t] - k_cum[key_t, start])
    return out


def add_history_features(history, targets, windows=DEFAULT_WINDOWS, smoothing=20.0):
    """Add rate_<level>_<w>d (shrunk late rate) and n_<level>_<w>d (trips behind it) to a
    copy of targets, computed only from history rows on earlier service days.

    history needs service_date, late and the level keys; targets needs service_date and keys.
    History and targets may be the same frame: each row still sees only earlier days.
    """
    out = targets.copy()
    parent = {w: np.full(len(targets), 0.5) for w in windows}
    for level, keys in LEVELS.items():
        counts = window_counts(history, targets, keys, windows)
        for w in windows:
            n, k = counts[w]
            rate = (k + smoothing * parent[w]) / (n + smoothing)
            out[f"rate_{level}_{w}d"] = rate.astype("float32")
            out[f"n_{level}_{w}d"] = n.astype("float32")
            parent[w] = rate
    return out


def history_feature_names(windows=DEFAULT_WINDOWS, counts=True):
    names = [f"rate_{level}_{w}d" for level in LEVELS for w in windows]
    if counts:
        names += [f"n_{level}_{w}d" for level in ("stop", "slot") for w in windows]
    return names
