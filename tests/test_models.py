import numpy as np
import pytest

pytest.importorskip("keras")

from traffic_flow import models  # noqa: E402


@pytest.mark.parametrize("build", [models.get_lstm, models.get_gru])
def test_recurrent_output_shape(build):
    m = build([12, 64, 64, 1])
    out = m.predict(np.zeros((5, 12, 1), dtype="float32"), verbose=0)
    assert out.shape == (5, 1)


def test_saes_returns_stages_and_final_model():
    stages = models.get_saes([12, 400, 400, 400, 1])
    assert len(stages) == 4
    out = stages[-1].predict(np.zeros((3, 12), dtype="float32"), verbose=0)
    assert out.shape == (3, 1)
    # names train_saes relies on when copying pre-trained weights
    for i in (1, 2, 3):
        stages[-1].get_layer("hidden%d" % i)
        stages[i - 1].get_layer("hidden")


def test_legacy_gru_uses_reset_after_false():
    # reset_after=True (Keras 3 default) has 2 bias vectors per gate set; legacy has 1.
    modern = models.get_gru([12, 64, 64, 1])
    legacy = models.get_gru([12, 64, 64, 1], legacy=True)
    assert modern.layers[0].get_weights()[2].shape == (2, 192)
    assert legacy.layers[0].get_weights()[2].shape == (192,)


def test_legacy_hard_sigmoid_formula():
    x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0], dtype="float32")
    y = np.asarray(models.legacy_hard_sigmoid(x))
    np.testing.assert_allclose(y, [0.0, 0.3, 0.5, 0.7, 1.0], atol=1e-6)
