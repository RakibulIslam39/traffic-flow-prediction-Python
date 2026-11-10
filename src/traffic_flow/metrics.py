"""Regression metrics used to compare the models."""
import math

import numpy as np
import sklearn.metrics as skm


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error in %, ignoring points where ``y_true <= 0``
    (same convention as the original project)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100)


def regression_report(y_true, y_pred):
    """All metrics as a dict."""
    mse = skm.mean_squared_error(y_true, y_pred)
    return {
        "mae": skm.mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mape": mape(y_true, y_pred),
        "r2": skm.r2_score(y_true, y_pred),
        "explained_variance": skm.explained_variance_score(y_true, y_pred),
    }
