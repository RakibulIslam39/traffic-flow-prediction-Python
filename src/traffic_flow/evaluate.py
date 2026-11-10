"""
Evaluate trained models on the test set:  python -m traffic_flow.evaluate

  --weights-dir artifacts          your own run (<name>.keras files)
  --legacy                         the original Keras 2.1 weights in artifacts/legacy
"""
import argparse
import os

import keras
import numpy as np
import pandas as pd

from traffic_flow.data import process_data, target_times
from traffic_flow.legacy import load_legacy_model
from traffic_flow.metrics import regression_report

NAMES = ["lstm", "gru", "saes"]
LABELS = {"lstm": "LSTM", "gru": "GRU", "saes": "SAEs"}


def load_models(weights_dir, legacy=False, lags=12):
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
        preds[name] = scaler.inverse_transform(p.reshape(-1, 1)).ravel()
    return y_true, preds


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
    parser.add_argument("--show", action="store_true", help="Also open the plot window.")
    args = parser.parse_args()

    weights_dir = args.weights_dir or ("artifacts/legacy" if args.legacy else "artifacts")
    try:
        loaded = load_models(weights_dir, legacy=args.legacy)
    except FileNotFoundError as err:
        parser.exit(1, "error: %s\n" % err)
    contiguous = not args.legacy_windows
    y_true, preds = predict_all(loaded, args.train_csv, args.test_csv,
                                contiguous=contiguous)

    rows = {LABELS[n]: regression_report(y_true, p) for n, p in preds.items()}
    table = pd.DataFrame(rows).T
    print(table.round(4).to_string())

    os.makedirs(args.out_dir, exist_ok=True)
    table.to_csv(os.path.join(args.out_dir, "metrics.csv"))
    times = target_times(args.test_csv, 12, contiguous=contiguous)
    plot_results(y_true, preds, os.path.join(args.out_dir, "comparison.png"),
                 times=times, show=args.show)
    print("saved metrics.csv and comparison.png to", args.out_dir)


if __name__ == "__main__":
    main()
