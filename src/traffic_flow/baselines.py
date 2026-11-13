"""
Simple baselines every neural model should be compared against.

All of them use exactly the same inputs as the neural models (the previous ``lags``
readings, scaled) and train in under a second.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

LABELS = {"persistence": "Persistence", "ridge": "Ridge", "gboost": "GradBoost"}


def persistence(X, lags=None):
    """Predict that the target equals the last flow reading.

    ``lags`` is the number of flow columns at the start of ``X``; by default every column is
    a flow reading. Pass it when ``X`` also carries time features after the flow columns.
    """
    X = np.asarray(X)
    return X[:, (lags if lags else X.shape[1]) - 1]


def fit_predict(X_train, y_train, X_eval, lags=None):
    """Fit each baseline on the training windows and predict ``X_eval`` (scaled units).

    Returns ``{name: predictions}`` for 'persistence', 'ridge' and 'gboost'.
    """
    preds = {"persistence": persistence(X_eval, lags)}
    preds["ridge"] = Ridge(alpha=1e-3).fit(X_train, y_train).predict(X_eval)
    gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
    preds["gboost"] = gb.fit(X_train, y_train).predict(X_eval)
    return preds
