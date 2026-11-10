"""
Load, scale and window the PeMS flow series.

Each sample is ``lags`` consecutive readings (the input) followed by the reading to predict.
By default a window is only created when all of its readings are 5 minutes apart, so no
sample spans a gap in the recording. ``contiguous=False`` reproduces the original project's
windowing, which ignores timestamps; it exists only to re-create the numbers in the old README.
"""
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ATTR = "Lane 1 Flow (Veh/5 Minutes)"
TIME_COL = "5 Minutes"
TIME_FORMAT = "%d/%m/%Y %H:%M"
INTERVAL = pd.Timedelta("5min")


def load_series(path):
    """Read a PeMS CSV and return ``(timestamps, flow)``. Missing flow is filled with 0
    (as in the original project; the raw files currently contain no NaNs)."""
    if not os.path.exists(path):
        raise FileNotFoundError("data file not found: %s (see data/README.md)" % path)
    df = pd.read_csv(path, encoding="utf-8")
    if ATTR not in df.columns or TIME_COL not in df.columns:
        raise ValueError("%s must contain the columns '%s' and '%s'" % (path, TIME_COL, ATTR))
    times = pd.to_datetime(df[TIME_COL], format=TIME_FORMAT)
    flow = df[ATTR].fillna(0).values.astype(float)
    return times, flow


def _window_starts(times, lags, contiguous):
    """Target positions ``i`` for which the window ``i-lags .. i`` is usable."""
    n = len(times)
    if not contiguous:
        return list(range(lags, n))
    times = pd.Series(pd.to_datetime(times)).reset_index(drop=True)
    # seg[i] = index of the first row of the contiguous segment containing row i
    breaks = np.asarray((times.diff() != INTERVAL).fillna(True), dtype=bool)
    seg = np.maximum.accumulate(np.where(breaks, np.arange(n), 0))
    return [i for i in range(lags, n) if i - lags >= seg[i]]


def _windows(flow, starts, lags):
    return np.array([flow[i - lags: i + 1] for i in starts])


def process_data(train, test, lags, contiguous=True):
    """Scale and window train/test data.

    The scaler is fit on the training flow only. Test windows keep chronological order.
    Training windows are shuffled (legacy behaviour, see README).

    Returns ``X_train, y_train, X_test, y_test, scaler``.
    """
    t_train, f_train = load_series(train)
    t_test, f_test = load_series(test)

    scaler = MinMaxScaler(feature_range=(0, 1)).fit(f_train.reshape(-1, 1))
    s_train = scaler.transform(f_train.reshape(-1, 1)).ravel()
    s_test = scaler.transform(f_test.reshape(-1, 1)).ravel()

    train_w = _windows(s_train, _window_starts(t_train, lags, contiguous), lags)
    test_w = _windows(s_test, _window_starts(t_test, lags, contiguous), lags)
    if len(train_w) == 0 or len(test_w) == 0:
        raise ValueError("no usable windows: check that the files have enough contiguous data")

    np.random.shuffle(train_w)

    return (train_w[:, :-1], train_w[:, -1], test_w[:, :-1], test_w[:, -1], scaler)


def target_times(test, lags, contiguous=True):
    """Timestamp of each test target, in the same order as ``process_data``'s ``y_test``."""
    times, _ = load_series(test)
    return times.iloc[_window_starts(times, lags, contiguous)].reset_index(drop=True)
