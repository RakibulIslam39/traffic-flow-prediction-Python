import json
import os

import numpy as np
import pandas as pd
import pytest

from traffic_flow import baselines
from traffic_flow.data import TIME_FEATURES, make_datasets, time_features

ATTR = "Lane 1 Flow (Veh/5 Minutes)"
LAGS = 12


def _csv(path, days, start="2016-01-04", skip_day=None):
    """Full days of 5-minute rows; flow = row number. ``skip_day`` drops one day (a gap)."""
    t = pd.date_range(start, periods=days * 288, freq="5min")
    df = pd.DataFrame({"5 Minutes": t, ATTR: np.arange(len(t), dtype=float)})
    if skip_day is not None:
        df = df[df["5 Minutes"].dt.normalize() != pd.Timestamp(start) + pd.Timedelta(days=skip_day)]
    df["5 Minutes"] = df["5 Minutes"].dt.strftime("%d/%m/%Y %H:%M")
    df.to_csv(path, index=False)


@pytest.fixture
def csvs(tmp_path):
    _csv(tmp_path / "train.csv", days=4)
    _csv(tmp_path / "test.csv", days=3, start="2016-02-01", skip_day=1)
    return str(tmp_path / "train.csv"), str(tmp_path / "test.csv")


def _raw(d, a):
    return d["scaler"].inverse_transform(np.asarray(a).reshape(-1, 1)).reshape(np.shape(a))


@pytest.mark.parametrize("h", [1, 3, 12])
def test_target_is_horizon_steps_after_the_last_input(csvs, h):
    d = make_datasets(*csvs, LAGS, val_days=1, horizon=h, shuffle=False)
    for split in ("train", "val", "test"):
        X, y = _raw(d, d["X_" + split]), _raw(d, d["y_" + split])
        np.testing.assert_allclose(np.diff(X, axis=1), 1.0, atol=1e-6)   # inputs consecutive
        np.testing.assert_allclose(y - X[:, -1], h, atol=1e-6)           # target h rows later


def test_longer_horizon_drops_windows_at_segment_starts(csvs):
    n1 = len(make_datasets(*csvs, LAGS, val_days=0, horizon=1)["y_test"])
    n12 = len(make_datasets(*csvs, LAGS, val_days=0, horizon=12)["y_test"])
    # the test file has 2 contiguous days: each loses lags + h - 1 targets at its start
    assert n1 == 2 * (288 - LAGS)
    assert n12 == 2 * (288 - LAGS - 11)


def test_horizon_must_be_positive(csvs):
    with pytest.raises(ValueError, match="horizon"):
        make_datasets(*csvs, LAGS, horizon=0)


def test_time_features_are_cyclical_and_bounded():
    t = pd.to_datetime(["2016-01-04 00:00", "2016-01-04 06:00", "2016-01-04 12:00",
                        "2016-01-04 23:55", "2016-01-05 00:00"])
    f = time_features(t)
    assert f.shape == (5, len(TIME_FEATURES))
    assert np.all(np.abs(f) <= 1.0)
    np.testing.assert_allclose(f[0, :2], [0.0, 1.0], atol=1e-9)      # midnight
    np.testing.assert_allclose(f[1, :2], [1.0, 0.0], atol=1e-9)      # 06:00
    np.testing.assert_allclose(f[2, :2], [0.0, -1.0], atol=1e-9)     # noon
    # 23:55 is next to 00:00 in feature space, far from noon
    assert np.linalg.norm(f[3, :2] - f[4, :2]) < 0.05 < np.linalg.norm(f[3, :2] - f[2, :2])
    # Monday and Tuesday differ in the day-of-week part
    assert not np.allclose(f[0, 2:], f[4, 2:])


def test_features_describe_the_target_time_and_follow_the_shuffle(csvs):
    d = make_datasets(*csvs, LAGS, val_days=0, horizon=3, features=True, shuffle=False)
    assert d["X_test"].shape[1] == LAGS + len(TIME_FEATURES)
    np.testing.assert_allclose(d["X_test"][:, LAGS:], time_features(d["test_times"]))
    # shuffled training rows keep each feature row attached to its own window:
    # flow = row number, so the target row's time of day can be recomputed from y
    np.random.seed(3)
    s = make_datasets(*csvs, LAGS, val_days=0, horizon=3, features=True, shuffle=True)
    y = _raw(s, s["y_train"])
    tod = (np.round(y) % 288) / 288 * 2 * np.pi
    np.testing.assert_allclose(s["X_train"][:, LAGS], np.sin(tod), atol=1e-6)


def test_features_off_by_default_and_test_set_independent_of_them(csvs):
    a = make_datasets(*csvs, LAGS, val_days=0, shuffle=False)
    b = make_datasets(*csvs, LAGS, val_days=0, shuffle=False, features=True)
    assert a["X_test"].shape[1] == LAGS
    np.testing.assert_array_equal(a["X_test"], b["X_test"][:, :LAGS])
    np.testing.assert_array_equal(a["y_test"], b["y_test"])


def test_persistence_uses_last_flow_column_when_features_follow():
    X = np.array([[1.0, 2.0, 3.0, 0.5, -0.5]])
    assert baselines.persistence(X, lags=3)[0] == 3.0
    assert baselines.persistence(X[:, :3])[0] == 3.0


# ---- model side (needs keras)
def test_recurrent_input_repeats_features_along_the_sequence():
    pytest.importorskip("keras")
    from traffic_flow import train

    X = np.hstack([np.arange(24, dtype=float).reshape(2, 12), [[0.1, 0.2], [0.3, 0.4]]])
    seq = train._for_model("lstm", X)
    assert seq.shape == (2, 12, 3)
    np.testing.assert_array_equal(seq[:, :, 0], X[:, :12])
    assert np.all(seq[0, :, 1] == 0.1) and np.all(seq[1, :, 2] == 0.4)
    assert train._for_model("saes", X).shape == (2, 14)
    assert train._for_model("gru", X[:, :12]).shape == (2, 12, 1)


def test_train_and_evaluate_with_features_and_horizon(csvs, tmp_path):
    pytest.importorskip("keras")
    from traffic_flow import train
    from traffic_flow.evaluate import load_models, predict_all

    d = make_datasets(*csvs, LAGS, val_days=1, horizon=3, features=True)
    out = str(tmp_path / "m")
    config = {"batch": 64, "epochs": 1, "out_dir": out, "verbose": 0,
              "horizon": 3, "features": True}
    for name in ("gru", "saes"):
        train.run(name, d["X_train"], d["y_train"], config, val=(d["X_val"], d["y_val"]))
    with open(os.path.join(out, "gru.json")) as f:
        assert json.load(f) == {"lags": 12, "horizon": 3, "features": True}

    loaded, meta = load_models(out)                 # lstm is absent: only two are loaded
    assert set(loaded) == {"gru", "saes"} and meta == {"horizon": 3, "features": True}
    y_true, preds = predict_all(loaded, *csvs, horizon=3, features=True)
    assert all(p.shape == y_true.shape for p in preds.values())


def test_load_models_rejects_mixed_settings_and_empty_folder(csvs, tmp_path):
    pytest.importorskip("keras")
    from traffic_flow import train
    from traffic_flow.evaluate import load_models

    with pytest.raises(FileNotFoundError, match="no .* models found"):
        load_models(str(tmp_path / "empty"))
    out = str(tmp_path / "m")
    for name, h in (("gru", 1), ("saes", 3)):
        d = make_datasets(*csvs, LAGS, val_days=0, horizon=h)
        train.run(name, d["X_train"], d["y_train"],
                  {"batch": 64, "epochs": 1, "out_dir": out, "verbose": 0, "horizon": h})
    with pytest.raises(ValueError, match="different settings"):
        load_models(out)
