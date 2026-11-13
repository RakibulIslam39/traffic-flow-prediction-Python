"""
Evaluate trained models on the test set:  python -m traffic_flow.evaluate

  --weights-dir artifacts          your own run (<name>.keras files)
  --legacy                         the original Keras 2.1 weights in artifacts/legacy
"""
import argparse
<<<<<<< HEAD
import json
=======
>>>>>>> origin/main
import os

import keras
import numpy as np
import pandas as pd

<<<<<<< HEAD
from traffic_flow import baselines, train
from traffic_flow.data import make_datasets
=======
from traffic_flow.data import process_data, target_times
>>>>>>> origin/main
from traffic_flow.legacy import load_legacy_model
from traffic_flow.metrics import regression_report

NAMES = ["lstm", "gru", "saes"]
LABELS = {"lstm": "LSTM", "gru": "GRU", "saes": "SAEs"}


def load_models(weights_dir, legacy=False, lags=12):
<<<<<<< HEAD
    """Load whichever of the three models exist in ``weights_dir``.

    Returns ``(models, meta)``. ``meta`` holds the ``horizon`` and ``features`` settings the
    models were trained with (read from the ``<name>.json`` files that ``train.py`` writes;
    models without one are taken to be 5-minute, flow-only models).
    """
    ext = ".h5" if legacy else ".keras"
    present = [n for n in NAMES if os.path.exists(os.path.join(weights_dir, n + ext))]
    if not present:
        hint = ("" if legacy else " Train them first: python -m traffic_flow.train --model all")
        raise FileNotFoundError("no %s models found in %s.%s"
                                % (" / ".join(n + ext for n in NAMES), weights_dir, hint))
    loaded, metas = {}, {}
    for n in present:
        path = os.path.join(weights_dir, n + ext)
        loaded[n] = load_legacy_model(n, path, lags) if legacy else keras.saving.load_model(path)
        meta = {"horizon": 1, "features": False}
        meta_path = os.path.join(weights_dir, n + ".json")
        if not legacy and os.path.exists(meta_path):
            with open(meta_path) as f:
                meta.update({k: v for k, v in json.load(f).items() if k in meta})
        metas[n] = meta
    distinct = {(m["horizon"], m["features"]) for m in metas.values()}
    if len(distinct) > 1:
        raise ValueError("models in %s were trained with different settings (%s); "
                         "keep each horizon/features combination in its own folder"
                         % (weights_dir, metas))
    return loaded, next(iter(metas.values()))


def predict_all(loaded, train_csv, test_csv, lags=12, contiguous=True, horizon=1,
                features=False):
    """Return ``(y_true, {name: y_pred})`` in original (un-scaled) units."""
    d = make_datasets(train_csv, test_csv, lags, val_days=0, contiguous=contiguous,
                      shuffle=False, horizon=horizon, features=features)
    scaler = d["scaler"]
    y_true = scaler.inverse_transform(d["y_test"].reshape(-1, 1)).ravel()
    preds = {}
    for name, model in loaded.items():
        p = model.predict(train._for_model(name, d["X_test"]), verbose=0)
=======
    """Load the three models from ``weights_dir``. Raises a clear error if any is missing."""
    ext = ".h5" if legacy else ".keras"
    missing = [n + ext for n in NAMES
               if not os.path.exists(os.path.join(weights_dir, n + ext))]
    if missing:
        hint = ("" if legacy else " Train them first: python -m traffic_flow.train --model all")
        raise FileNotFoundError("missing %s in %s.%s" % (", ".join(missing), weights_dir, hint))
    loaded = {}
    for n in NAMES:
        path = os.path.join(weights_dir, n + ext)
        if legacy:
            loaded[n] = load_legacy_model(n, path, lags)
        else:
            loaded[n] = keras.saving.load_model(path)
    return loaded


def predict_all(loaded, train_csv, test_csv, lags=12, contiguous=True):
    """Return ``(y_true, {name: y_pred})`` in original (un-scaled) units."""
    _, _, X_test, y_test, scaler = process_data(train_csv, test_csv, lags, contiguous)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    preds = {}
    for name, model in loaded.items():
        X = X_test if name == "saes" else X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        p = model.predict(X, verbose=0)
>>>>>>> origin/main
        preds[name] = scaler.inverse_transform(p.reshape(-1, 1)).ravel()
    return y_true, preds


<<<<<<< HEAD
def baseline_rows(train_csv, test_csv, lags=12, val_days=5, contiguous=True, horizon=1,
                  features=False):
    """Metrics of the simple baselines on the test set, fit on the same training days and
    the same inputs that ``train.py`` uses."""
    d = make_datasets(train_csv, test_csv, lags, val_days=val_days, contiguous=contiguous,
                      horizon=horizon, features=features)
    inv = lambda a: d["scaler"].inverse_transform(np.asarray(a).reshape(-1, 1)).ravel()  # noqa: E731
    y_true = inv(d["y_test"])
    preds = baselines.fit_predict(d["X_train"], d["y_train"], d["X_test"], lags=lags)
    return {baselines.LABELS[n]: regression_report(y_true, inv(p)) for n, p in preds.items()}


=======
>>>>>>> origin/main
def plot_results(y_true, preds, path, times, show=False):
    """Plot one day (the first day in the test set) of true vs predicted flow.

    Rows are picked by timestamp, not by position, so a gap in the recording cannot pull
    in points from another day or join two days with a line.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    times = pd.Series(pd.to_datetime(times)).reset_index(drop=True)
    day = times.iloc[0].normalize()
    mask = (times.dt.normalize() == day).values
    x = times[mask]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, np.asarray(y_true)[mask], label="True Data")
    for name, p in preds.items():
        ax.plot(x, np.asarray(p)[mask], label=LABELS[name])
    ax.legend()
    ax.grid(True)
    ax.set_title("Test set: %s" % day.strftime("%Y-%m-%d"))
    ax.set_xlabel("Time of Day")
    ax.set_ylabel("Flow")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", default=None)
    parser.add_argument("--legacy", action="store_true",
                        help="Load the original .h5 weights (default dir: artifacts/legacy).")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument("--out-dir", default="artifacts/eval",
                        help="Where metrics.csv and the comparison plot are written.")
    parser.add_argument("--legacy-windows", action="store_true",
                        help="Use the original (timestamp-unaware) windowing. Reproduces the "
                             "numbers in the original README.")
<<<<<<< HEAD
    parser.add_argument("--no-baselines", action="store_true",
                        help="Do not add the Persistence / Ridge / GradBoost rows.")
    parser.add_argument("--val-days", type=int, default=5,
                        help="Validation days excluded from the baselines' training data "
                             "(match the value used for train.py).")
=======
>>>>>>> origin/main
    parser.add_argument("--show", action="store_true", help="Also open the plot window.")
    args = parser.parse_args()

    weights_dir = args.weights_dir or ("artifacts/legacy" if args.legacy else "artifacts")
    try:
<<<<<<< HEAD
        loaded, meta = load_models(weights_dir, legacy=args.legacy)
    except (FileNotFoundError, ValueError) as err:
        parser.exit(1, "error: %s\n" % err)
    contiguous = not args.legacy_windows
    horizon, features = meta["horizon"], meta["features"]
    print("forecast horizon: %d min, time features: %s"
          % (5 * horizon, "on" if features else "off"))
    y_true, preds = predict_all(loaded, args.train_csv, args.test_csv,
                                contiguous=contiguous, horizon=horizon, features=features)

    rows = {LABELS[n]: regression_report(y_true, p) for n, p in preds.items()}
    if not args.no_baselines:
        rows.update(baseline_rows(args.train_csv, args.test_csv, val_days=args.val_days,
                                  contiguous=contiguous, horizon=horizon, features=features))
=======
        loaded = load_models(weights_dir, legacy=args.legacy)
    except FileNotFoundError as err:
        parser.exit(1, "error: %s\n" % err)
    contiguous = not args.legacy_windows
    y_true, preds = predict_all(loaded, args.train_csv, args.test_csv,
                                contiguous=contiguous)

    rows = {LABELS[n]: regression_report(y_true, p) for n, p in preds.items()}
>>>>>>> origin/main
    table = pd.DataFrame(rows).T
    print(table.round(4).to_string())

    os.makedirs(args.out_dir, exist_ok=True)
    table.to_csv(os.path.join(args.out_dir, "metrics.csv"))
<<<<<<< HEAD
    times = make_datasets(args.train_csv, args.test_csv, 12, val_days=0, shuffle=False,
                          contiguous=contiguous, horizon=horizon)["test_times"]
=======
    times = target_times(args.test_csv, 12, contiguous=contiguous)
>>>>>>> origin/main
    plot_results(y_true, preds, os.path.join(args.out_dir, "comparison.png"),
                 times=times, show=args.show)
    print("saved metrics.csv and comparison.png to", args.out_dir)


if __name__ == "__main__":
    main()
