import numpy as np
import pandas as pd
import pytest

from traffic_flow.data import process_data, target_times

ATTR = "Lane 1 Flow (Veh/5 Minutes)"
TIME = "5 Minutes"


def _write_plain(path, values, start="01/01/2016 0:00"):
    t = pd.date_range(pd.to_datetime(start, format="%d/%m/%Y %H:%M"),
                      periods=len(values), freq="5min")
    pd.DataFrame({TIME: t.strftime("%d/%m/%Y %H:%M"), ATTR: values}).to_csv(path, index=False)


def _write_with_gap(path, values, gap_at, gap_len):
    """Timestamps are continuous except that ``gap_len`` intervals are skipped before row ``gap_at``."""
    t = pd.date_range("2016-01-01 00:00", periods=len(values), freq="5min")
    t_list = list(t)
    shift = pd.Timedelta("5min") * gap_len
    t_list = t_list[:gap_at] + [x + shift for x in t_list[gap_at:]]
    pd.DataFrame({TIME: [x.strftime("%d/%m/%Y %H:%M") for x in t_list],
                  ATTR: values}).to_csv(path, index=False)


@pytest.fixture
def csvs(tmp_path):
    train = np.arange(0, 100, dtype=float)
    test = np.arange(50, 80, dtype=float)
    _write_plain(tmp_path / "train.csv", train)
    _write_plain(tmp_path / "test.csv", test)
    return str(tmp_path / "train.csv"), str(tmp_path / "test.csv")


def test_window_shapes(csvs):
    lags = 12
    X_train, y_train, X_test, y_test, _ = process_data(*csvs, lags)
    assert X_train.shape == (100 - lags, lags)
    assert y_train.shape == (100 - lags,)
    assert X_test.shape == (30 - lags, lags)
    assert y_test.shape == (30 - lags,)


def test_legacy_windowing_matches_original_count(csvs):
    X_train, _, X_test, _, _ = process_data(*csvs, 12, contiguous=False)
    assert len(X_train) == 100 - 12 and len(X_test) == 30 - 12


def test_scaler_is_fit_on_train_only(csvs):
    _, _, _, _, scaler = process_data(*csvs, 12)
    assert scaler.data_min_[0] == 0 and scaler.data_max_[0] == 99


def test_test_windows_are_chronological(csvs):
    lags = 12
    _, _, X_test, y_test, scaler = process_data(*csvs, lags)
    y = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    np.testing.assert_allclose(y, np.arange(50 + lags, 80), atol=1e-6)


def test_train_windows_are_shuffled_but_consistent(csvs):
    lags = 12
    X_train, y_train, _, _, scaler = process_data(*csvs, lags)
    X = scaler.inverse_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    y = scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
    np.testing.assert_allclose(y, X[:, -1] + 1, atol=1e-6)


def test_no_window_spans_a_timestamp_gap(tmp_path):
    values = np.arange(60, dtype=float)
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    _write_with_gap(train, values, gap_at=30, gap_len=5)
    _write_with_gap(test, values, gap_at=30, gap_len=5)
    lags = 12
    X_train, y_train, X_test, y_test, scaler = process_data(str(train), str(test), lags)
    # each input window must be 12 consecutive readings (no jump across the gap)
    for X in (X_train, X_test):
        raw = scaler.inverse_transform(X.reshape(-1, 1)).reshape(X.shape)
        np.testing.assert_allclose(np.diff(raw, axis=1), 1.0, atol=1e-6)
    # windows that would cross the gap are dropped: 60-12 minus the 12 crossing ones
    assert len(X_train) == (60 - lags) - lags


def target_times_align_with_targets(tmp_path):
    values = np.arange(60, dtype=float)
    test = tmp_path / "test.csv"
    _write_with_gap(test, values, gap_at=30, gap_len=5)
    times = target_times(str(test), 12)
    assert len(times) == 60 - 12 - 12
    # first target is the 13th row, i.e. one hour after the first timestamp
    assert times.iloc[0] == pd.Timestamp("2016-01-01 01:00")


def test_missing_columns_raise_clear_error(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"x": [1, 2, 3]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="must contain the columns"):
        process_data(str(bad), str(bad), 1)
