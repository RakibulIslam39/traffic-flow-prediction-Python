import numpy as np

from traffic_flow import baselines


def test_persistence_returns_last_input():
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_array_equal(baselines.persistence(X), [3.0, 6.0])


def test_fit_predict_shapes_and_names():
    rng = np.random.RandomState(0)
    X = rng.rand(300, 12)
    y = X[:, -1] * 0.9 + 0.05
    preds = baselines.fit_predict(X[:250], y[:250], X[250:])
    assert set(preds) == {"persistence", "ridge", "gboost"}
    assert all(p.shape == (50,) for p in preds.values())


def test_ridge_recovers_a_linear_rule():
    rng = np.random.RandomState(1)
    X = rng.rand(500, 12)
    y = 0.5 * X[:, -1] + 0.3 * X[:, -2] + 0.1
    preds = baselines.fit_predict(X[:400], y[:400], X[400:])
    assert np.abs(preds["ridge"] - y[400:]).max() < 0.01
    # and it beats persistence on data where persistence is not the right rule
    assert np.abs(preds["ridge"] - y[400:]).mean() < np.abs(preds["persistence"] - y[400:]).mean()


def test_gboost_is_deterministic():
    rng = np.random.RandomState(2)
    X, y = rng.rand(300, 12), rng.rand(300)
    a = baselines.fit_predict(X[:250], y[:250], X[250:])["gboost"]
    b = baselines.fit_predict(X[:250], y[:250], X[250:])["gboost"]
    np.testing.assert_array_equal(a, b)
