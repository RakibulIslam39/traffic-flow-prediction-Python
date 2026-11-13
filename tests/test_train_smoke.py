import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("keras")

from traffic_flow import train  # noqa: E402
from traffic_flow.data import make_datasets, process_data  # noqa: E402

ATTR = "Lane 1 Flow (Veh/5 Minutes)"


def _pems_csv(path, flow):
    times = pd.date_range("2016-01-04 00:00", periods=len(flow), freq="5min")
    pd.DataFrame({"5 Minutes": times.strftime("%d/%m/%Y %H:%M"), ATTR: flow}).to_csv(
        path, index=False)


@pytest.fixture
def tiny_data(tmp_path):
    t = np.arange(600)
    flow = 50 + 40 * np.sin(2 * np.pi * t / 288)
    _pems_csv(tmp_path / "train.csv", flow)
    _pems_csv(tmp_path / "test.csv", flow[:200])
    return str(tmp_path / "train.csv"), str(tmp_path / "test.csv")


@pytest.mark.parametrize("name", ["lstm", "gru", "saes"])
def test_one_epoch_trains_and_saves_keras_file(tiny_data, tmp_path, name):
    import keras

    X_train, y_train, _, _, _ = process_data(*tiny_data, 12)
    out = str(tmp_path / "out")
    config = {"batch": 64, "epochs": 1, "out_dir": out, "verbose": 0}
    train.run(name, X_train, y_train, config)

    assert os.path.exists(os.path.join(out, name + ".keras"))
    assert os.path.exists(os.path.join(out, name + "_loss.csv"))
    model = keras.saving.load_model(os.path.join(out, name + ".keras"))
    X = X_train if name == "saes" else X_train.reshape(*X_train.shape, 1)
    assert model.predict(X[:4], verbose=0).shape == (4, 1)


def test_training_with_chronological_validation_logs_val_loss(tiny_data, tmp_path):
    d = make_datasets(*tiny_data, 12, val_days=1)
    out = str(tmp_path / "out")
    config = {"batch": 64, "epochs": 1, "out_dir": out, "verbose": 0}
    hist = train.run("gru", d["X_train"], d["y_train"], config, val=(d["X_val"], d["y_val"]))
    assert "val_loss" in hist.history
    assert "val_loss" in pd.read_csv(os.path.join(out, "gru_loss.csv")).columns


def test_training_without_validation_has_no_val_loss(tiny_data, tmp_path):
    X_train, y_train, _, _, _ = process_data(*tiny_data, 12)
    config = {"batch": 64, "epochs": 1, "out_dir": str(tmp_path / "o"), "verbose": 0}
    hist = train.run("gru", X_train, y_train, config)
    assert "val_loss" not in hist.history


def test_early_stopping_stops_before_max_epochs_and_restores_best(tiny_data, tmp_path):
    import keras

    keras.utils.set_random_seed(0)
    d = make_datasets(*tiny_data, 12, val_days=1)
    config = {"batch": 64, "epochs": 200, "out_dir": str(tmp_path / "o"), "verbose": 0,
              "optimizer": "adam", "lr": 0.01, "patience": 3, "reduce_lr": True}
    hist = train.run("gru", d["X_train"], d["y_train"], config, val=(d["X_val"], d["y_val"]))
    val_loss = hist.history["val_loss"]
    assert len(val_loss) < 200                                  # stopped early
    # restore_best_weights: the saved model scores the best val_loss, not the last one
    X_val = d["X_val"].reshape(*d["X_val"].shape, 1)
    saved = keras.saving.load_model(str(tmp_path / "o" / "gru.keras"))
    mse = float(np.mean((saved.predict(X_val, verbose=0).ravel() - d["y_val"]) ** 2))
    assert mse == pytest.approx(min(val_loss), rel=1e-3)


def test_early_stopping_without_validation_is_rejected(tiny_data, tmp_path):
    X_train, y_train, _, _, _ = process_data(*tiny_data, 12)
    config = {"batch": 64, "epochs": 2, "out_dir": str(tmp_path / "o"), "verbose": 0,
              "patience": 2}
    with pytest.raises(ValueError, match="need a validation set"):
        train.run("gru", X_train, y_train, config)


@pytest.mark.parametrize("opt", ["rmsprop", "adam"])
def test_optimizer_and_learning_rate_are_applied(opt):
    o = train.make_optimizer({"optimizer": opt, "lr": 0.005})
    assert type(o).__name__.lower() == opt
    assert float(o.learning_rate) == pytest.approx(0.005)


def test_unknown_optimizer_is_rejected():
    with pytest.raises(ValueError, match="unknown optimizer"):
        train.make_optimizer({"optimizer": "sgdx"})


def test_legacy_gates_option_builds_legacy_layers(tiny_data, tmp_path):
    d = make_datasets(*tiny_data, 12, val_days=0)
    config = {"batch": 64, "epochs": 1, "out_dir": str(tmp_path / "o"), "verbose": 0,
              "legacy_gates": True}
    hist = train.run("gru", d["X_train"], d["y_train"], config)
    assert hist.model.layers[0].get_weights()[2].shape == (192,)   # reset_after=False
