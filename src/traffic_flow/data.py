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


def _window_starts(times, lags, contiguous, horizon=1):
    """Target positions ``i`` whose window is usable.

    A window is ``lags`` input readings followed, ``horizon`` steps after the last input, by
    the target at row ``i``. With ``contiguous`` every reading from the first input to the
    target must be 5 minutes apart.
    """
    n = len(times)
    span = lags + horizon - 1          # rows between the first input and the target
    if not contiguous:
        return list(range(span, n))
    times = pd.Series(pd.to_datetime(times)).reset_index(drop=True)
    # seg[i] = index of the first row of the contiguous segment containing row i
    breaks = np.asarray((times.diff() != INTERVAL).fillna(True), dtype=bool)
    seg = np.maximum.accumulate(np.where(breaks, np.arange(n), 0))
    return [i for i in range(span, n) if i - span >= seg[i]]


def _windows(flow, starts, lags, horizon=1):
    """Rows of ``lags`` inputs followed by the target (``lags + 1`` columns)."""
    if len(starts) == 0:
        return np.empty((0, lags + 1))
    return np.array([np.append(flow[i - horizon + 1 - lags: i - horizon + 1], flow[i])
                     for i in starts])


TIME_FEATURES = ["tod_sin", "tod_cos", "dow_sin", "dow_cos"]


def time_features(times):
    """Cyclical encoding of the time of day and the day of week of each timestamp.

    Sine/cosine pairs keep 23:55 next to 00:00 and Sunday next to Monday.
    """
    t = pd.Series(pd.to_datetime(times)).reset_index(drop=True)
    if len(t) == 0:
        return np.empty((0, len(TIME_FEATURES)))
    tod = (t.dt.hour * 60 + t.dt.minute).values / 1440.0 * 2 * np.pi
    dow = t.dt.dayofweek.values / 7.0 * 2 * np.pi
    return np.column_stack([np.sin(tod), np.cos(tod), np.sin(dow), np.cos(dow)])


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


def make_datasets(train, test, lags, val_days=5, contiguous=True, shuffle=True,
                  horizon=1, features=False):
    """Train / validation / test windows with a chronological validation split.

    The last ``val_days`` calendar days of the training file become the validation set, so
    validation windows never overlap training windows and always lie after them in time.
    ``val_days=0`` gives an empty validation set (train on everything).

    The scaler is fit on the whole training file (training + validation days) so that every
    model and baseline shares one scale; the test file never influences it.

    ``horizon`` is how many 5-minute steps ahead the target lies (1 = next reading,
    3 = 15 minutes, 12 = one hour). ``features=True`` appends the time-of-day and
    day-of-week encoding of the *target* time to each row of ``X``, after the ``lags``
    flow columns; that time is known in advance, so using it is not a leak.

    Returns a dict with ``X_train, y_train, X_val, y_val, X_test, y_test, scaler,
    val_dates`` (the validation days) and ``test_times`` (timestamp of each test target).
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    t_train, f_train = load_series(train)
    t_test, f_test = load_series(test)

    scaler = MinMaxScaler(feature_range=(0, 1)).fit(f_train.reshape(-1, 1))
    s_train = scaler.transform(f_train.reshape(-1, 1)).ravel()
    s_test = scaler.transform(f_test.reshape(-1, 1)).ravel()

    days = pd.Series(pd.to_datetime(t_train)).dt.normalize().reset_index(drop=True)
    unique_days = sorted(days.unique())
    if val_days < 0 or val_days >= len(unique_days):
        raise ValueError("val_days must be between 0 and %d" % (len(unique_days) - 1))
    val_set = set(unique_days[len(unique_days) - val_days:]) if val_days else set()
    is_val = days.isin(val_set).values

    starts = _window_starts(t_train, lags, contiguous, horizon)
    train_idx = [i for i in starts if not is_val[i]]
    val_idx = [i for i in starts if is_val[i]]
    test_idx = _window_starts(t_test, lags, contiguous, horizon)
    train_w = _windows(s_train, train_idx, lags, horizon)
    val_w = _windows(s_train, val_idx, lags, horizon)
    test_w = _windows(s_test, test_idx, lags, horizon)
    if len(train_w) == 0 or len(test_w) == 0:
        raise ValueError("no usable windows: check that the files have enough contiguous data")

    def inputs(windows, times, idx):
        if not features:
            return windows[:, :-1]
        return np.hstack([windows[:, :-1], time_features(times.iloc[idx])])

    X_train = inputs(train_w, t_train, train_idx)
    y_train = train_w[:, -1]
    if shuffle:
        order = np.random.permutation(len(X_train))
        X_train, y_train = X_train[order], y_train[order]

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": inputs(val_w, t_train, val_idx), "y_val": val_w[:, -1],
        "X_test": inputs(test_w, t_test, test_idx), "y_test": test_w[:, -1],
        "horizon": horizon, "features": bool(features),
        "scaler": scaler,
        "val_dates": [pd.Timestamp(d) for d in sorted(val_set)],
        "test_times": pd.Series(pd.to_datetime(t_test)).iloc[test_idx].reset_index(drop=True),
    }


def target_times(test, lags, contiguous=True):
    """Timestamp of each test target, in the same order as ``process_data``'s ``y_test``."""
    times, _ = load_series(test)
    return times.iloc[_window_starts(times, lags, contiguous)].reset_index(drop=True)
