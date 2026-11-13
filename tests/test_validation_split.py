import numpy as np
import pandas as pd
import pytest

from traffic_flow.data import make_datasets

ATTR = "Lane 1 Flow (Veh/5 Minutes)"
LAGS = 12


def _csv(path, days, start="2016-01-04", per_day=288):
    """``days`` consecutive full days; the flow value encodes the row number."""
    t = pd.date_range(start, periods=days * per_day, freq="5min")
    pd.DataFrame({"5 Minutes": t.strftime("%d/%m/%Y %H:%M"),
                  ATTR: np.arange(len(t), dtype=float)}).to_csv(path, index=False)


@pytest.fixture
def csvs(tmp_path):
    _csv(tmp_path / "train.csv", days=4)
    _csv(tmp_path / "test.csv", days=1, start="2016-02-01")
    return str(tmp_path / "train.csv"), str(tmp_path / "test.csv")


def _raw(d, key):
    return d["scaler"].inverse_transform(d[key].reshape(-1, 1)).ravel()


def test_validation_is_the_last_days_and_disjoint_from_training(csvs):
    d = make_datasets(*csvs, LAGS, val_days=1)
    y_train, y_val = _raw(d, "y_train"), _raw(d, "y_val")
    assert len(d["val_dates"]) == 1 and d["val_dates"][0] == pd.Timestamp("2016-01-07")
    # every validation target lies after every training target
    assert y_val.min() > y_train.max()
    assert len(np.intersect1d(np.round(y_train), np.round(y_val))) == 0
    # one full day of targets in validation, the rest in training
    assert len(y_val) == 288
    assert len(y_train) == 3 * 288 - LAGS


def test_validation_windows_stay_chronological(csvs):
    d = make_datasets(*csvs, LAGS, val_days=1)
    assert np.all(np.diff(_raw(d, "y_val")) > 0)
    assert np.all(np.diff(_raw(d, "y_test")) > 0)


def test_val_days_zero_uses_everything_for_training(csvs):
    d = make_datasets(*csvs, LAGS, val_days=0)
    assert len(d["X_val"]) == 0 and len(d["y_val"]) == 0
    assert len(d["X_train"]) == 4 * 288 - LAGS


def test_test_set_does_not_depend_on_val_days(csvs):
    a = make_datasets(*csvs, LAGS, val_days=0)
    b = make_datasets(*csvs, LAGS, val_days=2)
    np.testing.assert_array_equal(a["X_test"], b["X_test"])
    np.testing.assert_array_equal(a["y_test"], b["y_test"])
    assert len(a["test_times"]) == len(a["y_test"])


def test_scaler_ignores_the_test_file(csvs):
    d = make_datasets(*csvs, LAGS, val_days=1)
    assert d["scaler"].data_max_[0] == 4 * 288 - 1      # max of the training file only


@pytest.mark.parametrize("bad", [-1, 4, 10])
def test_invalid_val_days_raise(csvs, bad):
    with pytest.raises(ValueError, match="val_days"):
        make_datasets(*csvs, LAGS, val_days=bad)
