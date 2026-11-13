"""Regression test: the original Keras 2.1 weights must reproduce the metrics that the
original README reports, when loaded into the Keras 3 code base (legacy windowing)."""
import os

import pytest

pytest.importorskip("keras")

from traffic_flow.evaluate import load_models, predict_all  # noqa: E402
from traffic_flow.metrics import regression_report  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY = os.path.join(ROOT, "artifacts", "legacy")
TRAIN = os.path.join(ROOT, "data", "raw", "train.csv")
TEST = os.path.join(ROOT, "data", "raw", "test.csv")

# (MAE, MSE) from the original README
EXPECTED = {"lstm": (7.21, 98.05), "gru": (7.20, 99.32), "saes": (7.06, 92.08)}


@pytest.mark.skipif(not os.path.exists(os.path.join(LEGACY, "lstm.h5")),
                    reason="legacy weights not available")
def test_legacy_weights_reproduce_readme_metrics():
<<<<<<< HEAD
    loaded, _ = load_models(LEGACY, legacy=True)
=======
    loaded = load_models(LEGACY, legacy=True)
>>>>>>> origin/main
    # The original README numbers were produced with timestamp-unaware windowing.
    y_true, preds = predict_all(loaded, TRAIN, TEST, contiguous=False)
    for name, (mae, mse) in EXPECTED.items():
        r = regression_report(y_true, preds[name])
        assert r["mae"] == pytest.approx(mae, abs=0.01), name
        assert r["mse"] == pytest.approx(mse, abs=0.05), name
