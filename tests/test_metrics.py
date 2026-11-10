import numpy as np
import pytest

from traffic_flow.metrics import mape, regression_report


def _original_mape(y_true, y_pred):
    """The loop implementation from the original project, kept as a reference."""
    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]
    num = len(y_pred)
    sums = 0
    for i in range(num):
        sums += abs(y[i] - y_pred[i]) / y[i]
    return sums * (100 / num)


def test_mape_matches_original_implementation():
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 150, size=500).astype(float)
    y_pred = y_true + rng.normal(0, 5, size=500)
    assert mape(y_true, y_pred) == pytest.approx(_original_mape(y_true, y_pred))


def test_mape_ignores_zero_targets():
    assert mape([0, 10], [5, 11]) == pytest.approx(10.0)


def test_mape_all_zero_is_nan():
    assert np.isnan(mape([0, 0], [1, 2]))


def test_report_keys_and_perfect_prediction():
    r = regression_report([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert set(r) == {"mae", "mse", "rmse", "mape", "r2", "explained_variance"}
    assert r["mae"] == 0 and r["r2"] == pytest.approx(1.0)
