"""
A ready-to-use forecaster: recent readings in, forecasts for several horizons out.

It uses gradient boosting with time features, the most accurate model on this data in every
comparison (see README), with one model per horizon. Fitting takes about a second, and
TensorFlow is not needed.

    f = Forecaster.fit("data/raw/train.csv")
    f.predict([12 most recent 5-minute readings], last_time="2016-03-04 08:00")
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from traffic_flow.data import INTERVAL, _window_starts, _windows, load_series, time_features
from traffic_flow.metrics import regression_report

DEFAULT_HORIZONS = (1, 3, 6, 12)        # 5, 15, 30 and 60 minutes
DEFAULT_PATH = "artifacts/forecaster.joblib"


class Forecaster:
    def __init__(self, models, flow_min, flow_max, lags, info=None):
        self.models = dict(models)              # {horizon_steps: fitted regressor}
        self.flow_min, self.flow_max = float(flow_min), float(flow_max)
        self.lags = int(lags)
        self.info = dict(info or {})

    # ---------------------------------------------------------------- scaling / inputs
    def _scale(self, flow):
        return (np.asarray(flow, dtype=float) - self.flow_min) / (self.flow_max - self.flow_min)

    def _unscale(self, scaled):
        return np.asarray(scaled) * (self.flow_max - self.flow_min) + self.flow_min

    def _dataset(self, times, flow, horizon):
        """Inputs, targets (raw units) and target times for every usable window."""
        idx = _window_starts(times, self.lags, True, horizon)
        w = _windows(self._scale(flow), idx, self.lags, horizon)
        t = pd.Series(pd.to_datetime(times)).iloc[idx].reset_index(drop=True)
        X = np.hstack([w[:, :-1], time_features(t)])
        return X, np.asarray(flow, dtype=float)[idx], t

    # ---------------------------------------------------------------- fit / save / load
    @classmethod
    def fit(cls, train_csv="data/raw/train.csv", horizons=DEFAULT_HORIZONS, lags=12):
        """Fit one model per horizon on every day of ``train_csv``."""
        times, flow = load_series(train_csv)
        self = cls({}, flow.min(), flow.max(), lags)
        for h in horizons:
            X, y, _ = self._dataset(times, flow, h)
            if len(X) == 0:
                raise ValueError("not enough contiguous data in %s for horizon %d" % (train_csv, h))
            model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
            self.models[int(h)] = model.fit(X, self._scale(y))
        t = pd.Series(pd.to_datetime(times))
        self.info = {"train_csv": os.path.basename(train_csv), "train_rows": int(len(flow)),
                     "train_days": int(t.dt.normalize().nunique()),
                     "train_start": str(t.min()), "train_end": str(t.max()),
                     "model": "gradient boosting + time-of-day / day-of-week features"}
        return self

    def save(self, path=DEFAULT_PATH):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"models": self.models, "flow_min": self.flow_min, "flow_max": self.flow_max,
                     "lags": self.lags, "info": self.info}, path)
        return path

    @classmethod
    def load(cls, path=DEFAULT_PATH):
        d = joblib.load(path)
        return cls(d["models"], d["flow_min"], d["flow_max"], d["lags"], d["info"])

    @classmethod
    def load_or_fit(cls, path=DEFAULT_PATH, train_csv="data/raw/train.csv"):
        """Load a saved forecaster, or fit and save one if the file does not exist."""
        if os.path.exists(path):
            return cls.load(path)
        self = cls.fit(train_csv)
        self.save(path)
        return self

    @property
    def horizons(self):
        return sorted(self.models)

    # ---------------------------------------------------------------- predict
    def predict(self, readings, last_time):
        """Forecast from the ``lags`` most recent readings.

        ``readings``: flow values 5 minutes apart, oldest first; the last one was measured
        at ``last_time``. Returns one dict per horizon:
        ``{"horizon_minutes", "time", "flow"}``.
        """
        readings = np.asarray(readings, dtype=float).ravel()
        if len(readings) != self.lags:
            raise ValueError("expected %d readings, got %d" % (self.lags, len(readings)))
        if not np.all(np.isfinite(readings)) or np.any(readings < 0):
            raise ValueError("readings must be finite and non-negative")
        last_time = pd.Timestamp(last_time)
        out = []
        for h in self.horizons:
            target = last_time + INTERVAL * h
            x = np.hstack([self._scale(readings), time_features([target])[0]])[None, :]
            flow = float(self._unscale(self.models[h].predict(x))[0])
            out.append({"horizon_minutes": 5 * h, "time": target.strftime("%Y-%m-%d %H:%M"),
                        "flow": round(max(flow, 0.0), 2)})
        return out

    def predict_from_csv(self, csv):
        """Forecast from the last ``lags`` rows of a PeMS-style CSV."""
        times, flow = load_series(csv)
        if len(flow) < self.lags:
            raise ValueError("%s has fewer than %d rows" % (csv, self.lags))
        tail = pd.Series(pd.to_datetime(times)).iloc[-self.lags:]
        if (tail.diff().dropna() != INTERVAL).any():
            raise ValueError("the last %d rows of %s are not 5 minutes apart" % (self.lags, csv))
        return self.predict(flow[-self.lags:], tail.iloc[-1])

    # ---------------------------------------------------------------- backtest
    def predict_series(self, csv, horizon):
        """Walk through ``csv`` and forecast every possible target at ``horizon`` steps.
        Returns a DataFrame with ``time, actual, forecast``."""
        times, flow = load_series(csv)
        X, y, t = self._dataset(times, flow, horizon)
        pred = np.clip(self._unscale(self.models[horizon].predict(X)), 0, None)
        return pd.DataFrame({"time": t, "actual": y, "forecast": pred})

    def backtest(self, csv="data/raw/test.csv"):
        """Accuracy on ``csv`` for every horizon, next to two simple references."""
        times, flow = load_series(csv)
        rows = []
        for h in self.horizons:
            s = self.predict_series(csv, h)
            idx = _window_starts(times, self.lags, True, h)
            last_reading = np.asarray(flow, dtype=float)[np.array(idx) - h]
            r = regression_report(s["actual"], s["forecast"])
            p = regression_report(s["actual"], last_reading)
            rows.append({"horizon_minutes": 5 * h, "mae": r["mae"], "rmse": r["rmse"],
                         "mape": r["mape"], "persistence_mae": p["mae"], "n": len(s)})
        return pd.DataFrame(rows)
