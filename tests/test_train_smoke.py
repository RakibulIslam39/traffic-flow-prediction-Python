import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("keras")

from traffic_flow import train  # noqa: E402
from traffic_flow.data import process_data  # noqa: E402

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
