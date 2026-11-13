"""
Train a model:  python -m traffic_flow.train --model lstm|gru|saes|all

<<<<<<< HEAD
Default recipe: MSE loss, Adam (lr 3e-3, halved when val_loss stalls), batch 256, early
stopping on val_loss with the best weights restored. Validation uses the last ``--val-days``
days of the training file. ``--original-recipe`` gives the original project's settings
(RMSprop, fixed number of epochs), which need about ten times longer for a worse result.
"""
import argparse
import json
=======
Phase 2 keeps the original training recipe (MSE loss, RMSprop, batch 256, 600 epochs,
5% validation split on shuffled windows) so results stay comparable with the legacy
weights. The recipe itself is revisited in Phase 3.
"""
import argparse
>>>>>>> origin/main
import os

import keras
import numpy as np
import pandas as pd

from traffic_flow import models
<<<<<<< HEAD
from traffic_flow.data import make_datasets

LAGS = 12
# val_loss is MSE on [0, 1]-scaled flow, around 2.5e-3 for a trained model: 5e-6 is ~0.2%.
DEFAULT_MIN_DELTA = 5e-6


def _for_model(name, X):
    """Shape a 2-D input array for the model called ``name``.

    The first ``LAGS`` columns of ``X`` are the flow readings; any further columns are extra
    features of the target time (see ``data.make_datasets(features=True)``). SAEs take the
    row as it is. Recurrent models take ``(samples, LAGS, 1 + n_extra)``: the flow as the
    first channel and each extra feature repeated along the sequence.
    """
    X = np.asarray(X)
    if name == "saes":
        return X
    seq = X[:, :LAGS, None]
    extra = X[:, LAGS:]
    if extra.shape[1] == 0:
        return seq
    return np.concatenate([seq, np.repeat(extra[:, None, :], LAGS, axis=1)], axis=2)


def make_optimizer(config):
    """Optimizer from ``config['optimizer']`` ('rmsprop' | 'adam') and ``config['lr']``."""
    name = config.get("optimizer", "rmsprop")
    lr = config.get("lr") or 1e-3
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    if name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=lr)
    raise ValueError("unknown optimizer: %s" % name)


def make_callbacks(config, has_val):
    """Early stopping and learning-rate reduction, both driven by ``val_loss``.

    ``config['patience']`` > 0 turns early stopping on (the best weights are restored).
    An epoch only counts as an improvement if ``val_loss`` drops by more than
    ``config['min_delta']``; without that threshold the loss keeps creeping down by tiny
    amounts and training never stops.
    ``config['reduce_lr']`` halves the learning rate when ``val_loss`` stalls for a third of
    the patience. Neither can work without a validation set.
    """
    patience = config.get("patience", 0)
    if not patience and not config.get("reduce_lr"):
        return []
    if not has_val:
        raise ValueError("early stopping / reduce_lr need a validation set (val_days > 0)")
    callbacks = []
    if config.get("reduce_lr"):
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(2, (patience or 30) // 3),
            min_lr=1e-5))
    if patience:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True,
            min_delta=config.get("min_delta", DEFAULT_MIN_DELTA)))
    return callbacks


def _fit(model, X, y, config, val=None):
    model.compile(loss="mse", optimizer=make_optimizer(config), metrics=["mape"])
    has_val = val is not None and len(val[0]) > 0
    return model.fit(
        X, y,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_data=val if has_val else None,
        callbacks=make_callbacks(config, has_val),
        verbose=config.get("verbose", 2))


def train_model(model, X_train, y_train, name, config, val=None):
    """Compile, fit and save a single model (``<out_dir>/<name>.keras`` + loss CSV).

    ``val`` is an optional ``(X_val, y_val)`` pair. It must be a separate, later slice of
    the data (see ``make_datasets``); nothing is split off the training windows here.
    """
    hist = _fit(model, X_train, y_train, config, val)

    out_dir = config.get("out_dir", "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, name + ".keras"))
    # what the model expects, so evaluate.py can rebuild the same inputs
    meta = {"lags": LAGS, "horizon": config.get("horizon", 1),
            "features": bool(config.get("features", False))}
    with open(os.path.join(out_dir, name + ".json"), "w") as f:
        json.dump(meta, f)
=======
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
>>>>>>> origin/main
    pd.DataFrame(hist.history).to_csv(
        os.path.join(out_dir, name + "_loss.csv"), encoding="utf-8", index=False)
    return hist


<<<<<<< HEAD
def train_saes(stages, X_train, y_train, name, config, val=None):
    """Greedy layer-wise pre-training of the three SAE stages, then fine-tune the stack."""
    has_val = val is not None and len(val[0]) > 0
    temp = X_train
    temp_val = val[0] if has_val else None
=======
def train_saes(stages, X_train, y_train, name, config):
    """Greedy layer-wise pre-training of the three SAE stages, then fine-tune the stack."""
    temp = X_train
>>>>>>> origin/main
    for i in range(len(stages) - 1):
        if i > 0:
            prev = stages[i - 1]
            hidden = keras.Model(inputs=prev.inputs,
                                 outputs=prev.get_layer("hidden").output)
            temp = hidden.predict(temp, verbose=0)
<<<<<<< HEAD
            if has_val:
                temp_val = hidden.predict(temp_val, verbose=0)
        _fit(stages[i], temp, y_train, config, (temp_val, val[1]) if has_val else None)
=======

        stage = stages[i]
        stage.compile(loss="mse", optimizer="rmsprop", metrics=["mape"])
        stage.fit(temp, y_train, batch_size=config["batch"],
                  epochs=config["epochs"], validation_split=0.05,
                  verbose=config.get("verbose", 2))
>>>>>>> origin/main

    saes = stages[-1]
    for i in range(len(stages) - 1):
        weights = stages[i].get_layer("hidden").get_weights()
        saes.get_layer("hidden%d" % (i + 1)).set_weights(weights)

<<<<<<< HEAD
    return train_model(saes, X_train, y_train, name, config, val)


def run(name, X_train, y_train, config, val=None):
    """Train the model called ``name`` ('lstm' | 'gru' | 'saes').

    ``X_train`` (and ``val[0]``) are 2-D ``(samples, lags)`` arrays as produced by ``data``.
    """
    if name not in ("lstm", "gru", "saes"):
        raise ValueError("unknown model: %s" % name)
    if val is not None:
        val = (_for_model(name, val[0]), val[1])
    if name == "saes":
        stages = models.get_saes([X_train.shape[1], 400, 400, 400, 1])
        return train_saes(stages, X_train, y_train, name, config, val)
    build = models.get_lstm if name == "lstm" else models.get_gru
    model = build([LAGS, 64, 64, 1], legacy=bool(config.get("legacy_gates")),
                  n_features=1 + X_train.shape[1] - LAGS)
    return train_model(model, _for_model(name, X_train), y_train, name, config, val)


def add_training_args(parser):
    """Recipe options shared by ``train.py`` and ``experiment.py``."""
    parser.add_argument("--optimizer", choices=["rmsprop", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=3e-3, help="Initial learning rate.")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience in epochs on val_loss (0 = off).")
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA,
                        help="Smallest val_loss drop that counts as an improvement.")
    parser.add_argument("--no-reduce-lr", dest="reduce_lr", action="store_false",
                        help="Do not halve the learning rate when val_loss stalls.")
    parser.add_argument("--original-recipe", action="store_true",
                        help="The original project's recipe: RMSprop, lr 1e-3, no early "
                             "stopping, no learning-rate schedule. Overrides the options above.")
    parser.add_argument("--horizon", type=int, default=1,
                        help="Steps ahead to predict: 1 = 5 min, 3 = 15 min, 12 = 1 hour.")
    parser.add_argument("--features", action="store_true",
                        help="Add time-of-day and day-of-week inputs.")
    parser.add_argument("--legacy-gates", action="store_true",
                        help="LSTM/GRU with the Keras 2.1 gate behaviour (hard_sigmoid).")


def training_config(args):
    """Training options as a config dict. Without a validation set, early stopping and the
    learning-rate schedule are switched off (they have nothing to watch)."""
    config = {"batch": args.batch, "epochs": args.epochs, "optimizer": args.optimizer,
              "lr": args.lr, "patience": args.patience, "min_delta": args.min_delta,
              "reduce_lr": args.reduce_lr, "legacy_gates": args.legacy_gates,
              "horizon": args.horizon, "features": args.features}
    if args.original_recipe:
        config.update({"optimizer": "rmsprop", "lr": 1e-3, "patience": 0, "reduce_lr": False})
    if args.val_days == 0 and (config["patience"] or config["reduce_lr"]):
        print("note: --val-days 0 leaves no validation set, so early stopping and the "
              "learning-rate schedule are off; training runs for all %d epochs "
              "(about 80 is enough with the default recipe)." % args.epochs)
        config.update({"patience": 0, "reduce_lr": False})
    return config
=======
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
>>>>>>> origin/main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lstm",
                        choices=["lstm", "gru", "saes", "all"], help="Model to train.")
<<<<<<< HEAD
    parser.add_argument("--epochs", type=int, default=600,
                        help="Maximum number of epochs (early stopping usually ends sooner).")
=======
    parser.add_argument("--epochs", type=int, default=600)
>>>>>>> origin/main
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument("--out-dir", default="artifacts",
                        help="Where weights (.keras) and loss histories are saved.")
<<<<<<< HEAD
    add_training_args(parser)
    parser.add_argument("--val-days", type=int, default=5,
                        help="Last N days of the training file held out for validation "
                             "(0 = train on everything, no validation).")
    parser.add_argument("--quiet", action="store_true", help="Hide per-epoch progress output.")
    args = parser.parse_args()

    config = training_config(args)
    config.update({"out_dir": args.out_dir, "verbose": 0 if args.quiet else 2})
    names = ["lstm", "gru", "saes"] if args.model == "all" else [args.model]
    for name in names:
        keras.utils.set_random_seed(args.seed)     # python, numpy and backend RNGs
        d = make_datasets(args.train_csv, args.test_csv, LAGS, val_days=args.val_days,
                          horizon=args.horizon, features=args.features)
        run(name, d["X_train"], d["y_train"], config, val=(d["X_val"], d["y_val"]))
=======
    parser.add_argument("--quiet", action="store_true", help="Hide per-epoch progress output.")
    args = parser.parse_args()

    config = {"batch": args.batch, "epochs": args.epochs, "out_dir": args.out_dir,
              "verbose": 0 if args.quiet else 2}
    names = ["lstm", "gru", "saes"] if args.model == "all" else [args.model]
    for name in names:
        keras.utils.set_random_seed(args.seed)     # python, numpy and backend RNGs
        X_train, y_train, _, _, _ = process_data(args.train_csv, args.test_csv, LAGS)
        run(name, X_train, y_train, config)
>>>>>>> origin/main


if __name__ == "__main__":
    main()
