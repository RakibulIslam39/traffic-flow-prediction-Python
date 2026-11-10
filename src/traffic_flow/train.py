"""
Train a model:  python -m traffic_flow.train --model lstm|gru|saes|all

Phase 2 keeps the original training recipe (MSE loss, RMSprop, batch 256, 600 epochs,
5% validation split on shuffled windows) so results stay comparable with the legacy
weights. The recipe itself is revisited in Phase 3.
"""
import argparse
import os

import keras
import numpy as np
import pandas as pd

from traffic_flow import models
from traffic_flow.data import process_data

LAGS = 12


def train_model(model, X_train, y_train, name, config):
    """Compile, fit and save a single model (``<out_dir>/<name>.keras`` + loss CSV)."""
    model.compile(loss="mse", optimizer="rmsprop", metrics=["mape"])
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05,
        verbose=config.get("verbose", 2))

    out_dir = config.get("out_dir", "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, name + ".keras"))
    pd.DataFrame(hist.history).to_csv(
        os.path.join(out_dir, name + "_loss.csv"), encoding="utf-8", index=False)
    return hist


def train_saes(stages, X_train, y_train, name, config):
    """Greedy layer-wise pre-training of the three SAE stages, then fine-tune the stack."""
    temp = X_train
    for i in range(len(stages) - 1):
        if i > 0:
            prev = stages[i - 1]
            hidden = keras.Model(inputs=prev.inputs,
                                 outputs=prev.get_layer("hidden").output)
            temp = hidden.predict(temp, verbose=0)

        stage = stages[i]
        stage.compile(loss="mse", optimizer="rmsprop", metrics=["mape"])
        stage.fit(temp, y_train, batch_size=config["batch"],
                  epochs=config["epochs"], validation_split=0.05,
                  verbose=config.get("verbose", 2))

    saes = stages[-1]
    for i in range(len(stages) - 1):
        weights = stages[i].get_layer("hidden").get_weights()
        saes.get_layer("hidden%d" % (i + 1)).set_weights(weights)

    return train_model(saes, X_train, y_train, name, config)


def run(name, X_train, y_train, config):
    """Train the model called ``name`` ('lstm' | 'gru' | 'saes')."""
    if name in ("lstm", "gru"):
        X = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        build = models.get_lstm if name == "lstm" else models.get_gru
        return train_model(build([LAGS, 64, 64, 1]), X, y_train, name, config)
    if name == "saes":
        stages = models.get_saes([LAGS, 400, 400, 400, 1])
        return train_saes(stages, X_train, y_train, name, config)
    raise ValueError("unknown model: %s" % name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lstm",
                        choices=["lstm", "gru", "saes", "all"], help="Model to train.")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument("--out-dir", default="artifacts",
                        help="Where weights (.keras) and loss histories are saved.")
    parser.add_argument("--quiet", action="store_true", help="Hide per-epoch progress output.")
    args = parser.parse_args()

    config = {"batch": args.batch, "epochs": args.epochs, "out_dir": args.out_dir,
              "verbose": 0 if args.quiet else 2}
    names = ["lstm", "gru", "saes"] if args.model == "all" else [args.model]
    for name in names:
        keras.utils.set_random_seed(args.seed)     # python, numpy and backend RNGs
        X_train, y_train, _, _, _ = process_data(args.train_csv, args.test_csv, LAGS)
        run(name, X_train, y_train, config)


if __name__ == "__main__":
    main()
