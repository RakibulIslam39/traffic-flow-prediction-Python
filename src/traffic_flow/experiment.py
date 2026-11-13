"""
Multi-seed experiment:  python -m traffic_flow.experiment --seeds 1 2 3 --epochs 600

Trains every requested model once per seed, scores each run on the validation days and on
the test set, and compares them with the baselines. One seed is one draw from a noisy
process, so results are reported as mean +/- standard deviation across seeds.

The run is resumable: finished (model, seed) pairs are stored in ``<out-dir>/runs.csv`` and
skipped when the command is started again.
"""
import argparse
import os
import time

import keras
import numpy as np
import pandas as pd

from traffic_flow import baselines, train
from traffic_flow.data import make_datasets
from traffic_flow.metrics import regression_report

LAGS = 12
METRICS = ["mae", "rmse", "mape", "r2"]


def _inv(scaler, a):
    return scaler.inverse_transform(np.asarray(a).reshape(-1, 1)).ravel()


def _score(scaler, y_scaled, pred_scaled, prefix):
    if len(y_scaled) == 0:
        return {}
    r = regression_report(_inv(scaler, y_scaled), _inv(scaler, pred_scaled))
    return {"%s_%s" % (prefix, k): r[k] for k in METRICS}


def run_one(name, seed, args):
    """Train ``name`` with ``seed`` and return one result row."""
    keras.utils.set_random_seed(seed)
    d = make_datasets(args.train_csv, args.test_csv, LAGS, val_days=args.val_days,
                      horizon=args.horizon, features=args.features)
    out_dir = os.path.join(args.out_dir, "%s_%s_seed%d" % (args.tag, name, seed))
    config = train.training_config(args)
    config.update({"out_dir": out_dir, "verbose": 0})
    start = time.time()
    hist = train.run(name, d["X_train"], d["y_train"], config, val=(d["X_val"], d["y_val"]))
    seconds = time.time() - start
    model = hist.model

    val_loss = hist.history.get("val_loss")
    row = {"model": name, "seed": seed, "epochs": args.epochs, "tag": args.tag,
           "horizon": args.horizon, "features": args.features,
           "epochs_run": len(hist.history["loss"]),
           "best_epoch": int(np.argmin(val_loss)) + 1 if val_loss else None,
           "seconds": round(seconds, 1),
           "train_loss": float(hist.history["loss"][-1])}
    for split in ("val", "test"):
        X, y = d["X_" + split], d["y_" + split]
        if len(X):
            pred = model.predict(train._for_model(name, X), verbose=0)
            row.update(_score(d["scaler"], y, pred, split))
    return row


def baseline_table(args):
    """Baselines on the same split (deterministic, so no seeds)."""
    np.random.seed(0)
    d = make_datasets(args.train_csv, args.test_csv, LAGS, val_days=args.val_days,
                      horizon=args.horizon, features=args.features)
    rows = []
    for split in ("val", "test"):
        if not len(d["X_" + split]):
            continue
        preds = baselines.fit_predict(d["X_train"], d["y_train"], d["X_" + split], lags=LAGS)
        for n, p in preds.items():
            row = next((r for r in rows if r["model"] == baselines.LABELS[n]), None)
            if row is None:
                row = {"model": baselines.LABELS[n]}
                rows.append(row)
            row.update(_score(d["scaler"], d["y_" + split], p, split))
    return pd.DataFrame(rows).set_index("model")


def summarise(runs):
    """Mean and standard deviation across seeds, one row per model."""
    cols = [c for c in runs.columns if c.split("_", 1)[0] in ("val", "test")]
    cols += [c for c in ("epochs_run", "seconds") if c in runs.columns]
    g = runs.groupby("model")[cols]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=0).add_suffix("_std"))
    out.insert(0, "seeds", runs.groupby("model")["seed"].nunique())
    return out[["seeds"] + sorted(c for c in out.columns if c != "seeds")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["lstm", "gru", "saes"],
                        choices=["lstm", "gru", "saes"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--val-days", type=int, default=5)
    parser.add_argument("--tag", default="default",
                        help="Name for this recipe; runs with different tags are kept apart.")
    train.add_training_args(parser)
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument("--out-dir", default="artifacts/experiment")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs_path = os.path.join(args.out_dir, "runs.csv")
    runs = pd.read_csv(runs_path) if os.path.exists(runs_path) else pd.DataFrame()

    for seed in args.seeds:
        for name in args.models:
            done = (len(runs) and ((runs["model"] == name) & (runs["seed"] == seed)
                                   & (runs["epochs"] == args.epochs)
                                   & (runs["tag"] == args.tag)).any())
            if done:
                print("skip %s seed %d (already in runs.csv)" % (name, seed), flush=True)
                continue
            row = run_one(name, seed, args)
            runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
            runs.to_csv(runs_path, index=False)
            print("done %s seed %d: test MAE %.3f" % (name, seed, row["test_mae"]), flush=True)

    runs = runs[(runs["epochs"] == args.epochs) & (runs["tag"] == args.tag)]
    runs = runs[runs["model"].isin(args.models)]
    summary = summarise(runs)
    base = baseline_table(args)
    summary.to_csv(os.path.join(args.out_dir, "summary_%s.csv" % args.tag))
    base.to_csv(os.path.join(args.out_dir, "baselines_%s.csv" % args.tag))

    show = ["test_mae", "test_rmse", "test_mape", "val_mae", "epochs_run", "seconds"]
    print("\nNeural models [%s], mean +/- std over seeds (max %d epochs):"
          % (args.tag, args.epochs))
    for model, r in summary.iterrows():
        cells = ["%s %.3f +/- %.3f" % (m, r[m + "_mean"], r[m + "_std"])
                 for m in show if m + "_mean" in r]
        print("  %-5s (%d seeds)  %s" % (model, r["seeds"], "   ".join(cells)))
    print("\nBaselines (deterministic, same inputs as the neural models):")
    print(base[[c for c in show if c in base.columns]].round(3).to_string())
    print("\nsaved runs.csv, summary_%s.csv and baselines_%s.csv to %s"
          % (args.tag, args.tag, args.out_dir))


if __name__ == "__main__":
    main()
