"""Model builders for src/train.py. Kept in their own module so MLflow (skops) can load
the saved pipelines: these names are passed to log_model as trusted types."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TRUSTED_TYPES = ["src.models.HistoryRate", "src.models.as_str", "src.models.numeric_transform",
                 "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
                 # HistGradientBoosting stores functools.partial(check_array, ...) for its input check.
                 "functools.partial", "sklearn.utils.validation.check_array"]


class HistoryRate(BaseEstimator, ClassifierMixin):
    """No training: the probability is one history column, e.g. the slot's 8-week late rate."""

    def __init__(self, column):
        self.column = column

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = X[self.column].to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


def as_str(X):
    return X.astype(str)


def numeric_transform(X):
    """Rates to log-odds, counts to log1p: both closer to linear in the late log-odds."""
    out = X.astype(float).copy()
    for c in out.columns:
        if c.startswith("rate_"):
            p = out[c].clip(1e-3, 1 - 1e-3)
            out[c] = np.log(p / (1 - p))
        elif c.startswith("n_"):
            out[c] = np.log1p(out[c])
    return out.fillna(0.0)


def build_model(cfg, categorical, numeric):
    kind, params = cfg["model"]["type"], cfg["model"].get("params", {})
    if kind == "majority":
        return DummyClassifier(strategy="most_frequent")
    if kind == "history_rate":
        return HistoryRate(**params)
    if kind == "logreg":
        parts = [("onehot", Pipeline([("str", FunctionTransformer(as_str)),
                                      ("enc", OneHotEncoder(handle_unknown="ignore"))]), categorical)]
        if numeric:
            parts.append(("numeric", Pipeline([("tf", FunctionTransformer(numeric_transform)),
                                               ("scale", StandardScaler())]), numeric))
        return Pipeline([("encode", ColumnTransformer(parts)), ("model", LogisticRegression(**params))])
    if kind == "hgb":
        # Gradient-boosted trees. Columns with pandas category dtype (route, direction, point
        # type, standard type, day type) are split natively; everything else is numeric.
        # Fixed number of iterations: no early stopping, so no random holdout inside train.
        return HistGradientBoostingClassifier(categorical_features="from_dtype", early_stopping=False,
                                              random_state=0, **params)
    raise ValueError(f"unknown model type: {kind}")
