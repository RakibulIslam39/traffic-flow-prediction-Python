"""
Step 4 experiment on PeMS08: does more data help the neural model catch gradient boosting?

  A. Learning curve  - one sensor at a time, 10 / 20 / 40 training days.
  B. Pooled training - one model trained on several sensors at once, tested per sensor.

Every model gets the time features. The last 15 days are the test set; the 5 days before
the test set are validation; the N days before that are training. Results are appended to
<out>/results.csv and finished rows are skipped, so the script can be restarted.

    scripts/download_pems08.sh
    python scripts/step4_experiment.py
"""
import argparse
import os
import tempfile
import time

import numpy as np
import pandas as pd

from traffic_flow import baselines
from traffic_flow.data import ATTR, TIME_COL, TIME_FORMAT, make_datasets
from traffic_flow.metrics import regression_report

LAGS = 12
START = "2016-07-01 00:00"      # PeMS08 has no timestamps; its weekly pattern fits this start
TEST_DAYS, VAL_DAYS = 15, 5
KEYS = ["scope", "sensor", "train_days", "horizon", "model"]


def sensor_csvs(flow, sensor, train_days, folder):
    """Write PeMS-style train/test CSVs for one sensor and return their paths."""
    times = pd.date_range(START, periods=flow.shape[0], freq="5min")
    df = pd.DataFrame({TIME_COL: times.strftime(TIME_FORMAT), ATTR: flow[:, sensor]})
    n = flow.shape[0]
    test_start = n - TEST_DAYS * 288
    train_start = test_start - (train_days + VAL_DAYS) * 288
    if train_start < 0:
        raise ValueError("not enough days for %d training days" % train_days)
    tr = os.path.join(folder, "train_%d_%d.csv" % (sensor, train_days))
    te = os.path.join(folder, "test_%d.csv" % sensor)
    df.iloc[train_start:test_start].to_csv(tr, index=False)
    df.iloc[test_start:].to_csv(te, index=False)
    return tr, te


def dataset(flow, sensor, train_days, horizon, folder):
    np.random.seed(0)
    tr, te = sensor_csvs(flow, sensor, train_days, folder)
    d = make_datasets(tr, te, LAGS, val_days=VAL_DAYS, horizon=horizon, features=True)
    # time-of-day average of the training days, split into weekday / weekend
    full = pd.read_csv(tr)
    t = pd.to_datetime(full[TIME_COL], format=TIME_FORMAT)
    keep = (t.dt.normalize() < min(d["val_dates"])).values
    key = lambda s: (s.dt.dayofweek >= 5).astype(int) * 10000 + s.dt.hour * 60 + s.dt.minute  # noqa: E731
    prof = full[ATTR][keep].groupby(key(t)[keep]).mean()
    d["profile_pred"] = prof.reindex(key(d["test_times"]).values).values
    return d


def inv(d, a):
    return d["scaler"].inverse_transform(np.asarray(a).reshape(-1, 1)).ravel()


def score(d, pred_raw):
    r = regression_report(inv(d, d["y_test"]), pred_raw)
    return {"mae": r["mae"], "rmse": r["rmse"], "mape": r["mape"]}


def train_lstm(X, y, Xv, yv, batch, seed, out_dir, horizon):
    import keras
    from traffic_flow import train
    keras.utils.set_random_seed(seed)
    config = {"batch": batch, "epochs": 600, "optimizer": "adam", "lr": 3e-3, "patience": 20,
              "reduce_lr": True, "out_dir": out_dir, "verbose": 0, "horizon": horizon,
              "features": True}
    start = time.time()
    hist = train.run("lstm", X, y, config, val=(Xv, yv))
    return hist.model, len(hist.history["loss"]), time.time() - start


class Results:
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

    def has(self, **key):
        if not len(self.df):
            return False
        m = np.ones(len(self.df), dtype=bool)
        for k, v in key.items():
            m &= (self.df[k] == v).values
        return bool(m.any())

    def add(self, row):
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self.df.to_csv(self.path, index=False)
        print("done %(scope)s sensor %(sensor)s days %(train_days)s h%(horizon)s %(model)s: "
              "MAE %(mae).2f" % row, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="data/pems08/PEMS08.npz")
    p.add_argument("--out", default="artifacts/step4")
    p.add_argument("--sensors", nargs="+", type=int, default=[0, 100])
    p.add_argument("--pool", nargs="+", type=int, default=[0, 25, 50, 75, 100, 125])
    p.add_argument("--sizes", nargs="+", type=int, default=[10, 20, 40])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--skip-lstm", action="store_true")
    args = p.parse_args()

    from traffic_flow import train
    flow = np.load(args.npz)["data"][:, :, 0]
    os.makedirs(args.out, exist_ok=True)
    res = Results(os.path.join(args.out, "results.csv"))
    tmp = tempfile.mkdtemp()
    big = max(args.sizes)

    # ---- baselines everywhere (seconds)
    for sensor in args.sensors:
        for days in args.sizes:
            for h in (1, 12):
                key = dict(scope="single", sensor=sensor, train_days=days, horizon=h)
                if res.has(model="GradBoost", **key):
                    continue
                d = dataset(flow, sensor, days, h, tmp)
                preds = baselines.fit_predict(d["X_train"], d["y_train"], d["X_test"], lags=LAGS)
                for n, pr in preds.items():
                    res.add(dict(key, model=baselines.LABELS[n], **score(d, inv(d, pr))))
                res.add(dict(key, model="TimeOfDayAvg", **score(d, d["profile_pred"])))

    # ---- pooled gradient boosting, then LSTM runs
    def pooled(h):
        ds = {s: dataset(flow, s, big, h, tmp) for s in sorted(set(args.pool) | set(args.sensors))}
        cat = lambda k: np.concatenate([ds[s][k] for s in ds])  # noqa: E731
        return ds, cat("X_train"), cat("y_train"), cat("X_val"), cat("y_val")

    for h in (1, 12):
        if not res.has(scope="pooled", model="GradBoost", horizon=h, sensor=args.sensors[0]):
            ds, X, y, Xv, yv = pooled(h)
            from sklearn.ensemble import HistGradientBoostingRegressor
            gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                               random_state=0).fit(X, y)
            for s in args.sensors:
                res.add(dict(scope="pooled", sensor=s, train_days=big, horizon=h,
                             model="GradBoost", **score(ds[s], inv(ds[s], gb.predict(ds[s]["X_test"])))))
    if args.skip_lstm:
        return

    def single_lstm(sensor, days, h):
        key = dict(scope="single", sensor=sensor, train_days=days, horizon=h, model="LSTM")
        if res.has(**key):
            return
        d = dataset(flow, sensor, days, h, tmp)
        model, epochs, secs = train_lstm(d["X_train"], d["y_train"], d["X_val"], d["y_val"],
                                         256, args.seed, os.path.join(args.out, "m"), h)
        pred = inv(d, model.predict(train._for_model("lstm", d["X_test"]), verbose=0))
        res.add(dict(key, epochs_run=epochs, seconds=round(secs, 1), **score(d, pred)))

    for days in args.sizes:
        for sensor in args.sensors:
            single_lstm(sensor, days, 1)

    if not res.has(scope="pooled", model="LSTM", horizon=1, sensor=args.sensors[0]):
        ds, X, y, Xv, yv = pooled(1)
        order = np.random.RandomState(0).permutation(len(X))
        model, epochs, secs = train_lstm(X[order], y[order], Xv, yv, 1024, args.seed,
                                         os.path.join(args.out, "m_pooled"), 1)
        for s in args.sensors:
            pred = inv(ds[s], model.predict(train._for_model("lstm", ds[s]["X_test"]), verbose=0))
            res.add(dict(scope="pooled", sensor=s, train_days=big, horizon=1, model="LSTM",
                         epochs_run=epochs, seconds=round(secs, 1), **score(ds[s], pred)))

    for sensor in args.sensors:
        single_lstm(sensor, big, 12)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
