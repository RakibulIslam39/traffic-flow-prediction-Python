"""
Model definitions (Keras 3).

The architectures are the same as the original project:

* LSTM / GRU: two stacked recurrent layers (64, 64) -> Dropout(0.2) -> Dense(1, sigmoid)
* SAEs: three stacked Dense(400) layers -> Dropout(0.2) -> Dense(1, sigmoid),
  with greedy layer-wise pre-training (see ``train.py``).

Use ``legacy=True`` to rebuild the exact Keras 2.1 flavour of the recurrent layers
(``hard_sigmoid`` gates, ``reset_after=False`` for GRU). That is only needed to load
the original ``artifacts/legacy/*.h5`` weights, see ``legacy.py``.
"""
import keras
from keras import layers


@keras.saving.register_keras_serializable(package="traffic_flow")
def legacy_hard_sigmoid(x):
    """Keras 2.1 ``hard_sigmoid``: clip(0.2 * x + 0.5, 0, 1).

    Keras 3 changed the formula of its built-in ``hard_sigmoid`` (slope 1/6), so the
    original gate activation has to be written out explicitly.
    """
    return keras.ops.clip(0.2 * x + 0.5, 0.0, 1.0)


def _recurrent_kwargs(cell, legacy):
    if not legacy:
        return {}
    kwargs = {"recurrent_activation": legacy_hard_sigmoid}
    if cell == "gru":
        kwargs["reset_after"] = False
    return kwargs


def _get_rnn(cell, units, legacy):
    layer = layers.LSTM if cell == "lstm" else layers.GRU
    kw = _recurrent_kwargs(cell, legacy)
    model = keras.Sequential(name=cell)
    model.add(keras.Input(shape=(units[0], 1)))
    model.add(layer(units[1], return_sequences=True, **kw))
    model.add(layer(units[2], **kw))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units[3], activation="sigmoid"))
    return model


def get_lstm(units, legacy=False):
    """LSTM model. ``units`` = [n_inputs(lags), hidden1, hidden2, n_outputs]."""
    return _get_rnn("lstm", units, legacy)


def get_gru(units, legacy=False):
    """GRU model. ``units`` = [n_inputs(lags), hidden1, hidden2, n_outputs]."""
    return _get_rnn("gru", units, legacy)


def _get_sae(inputs, hidden, output):
    """One auto-encoder stage used for greedy layer-wise pre-training."""
    model = keras.Sequential()
    model.add(keras.Input(shape=(inputs,)))
    model.add(layers.Dense(hidden, name="hidden"))
    model.add(layers.Activation("sigmoid"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(output, activation="sigmoid"))
    return model


def get_saes(layer_sizes):
    """Stacked auto-encoders.

    ``layer_sizes`` = [n_inputs, h1, h2, h3, n_outputs].
    Returns ``[sae1, sae2, sae3, saes]``: three pre-training stages and the final model.
    """
    sae1 = _get_sae(layer_sizes[0], layer_sizes[1], layer_sizes[-1])
    sae2 = _get_sae(layer_sizes[1], layer_sizes[2], layer_sizes[-1])
    sae3 = _get_sae(layer_sizes[2], layer_sizes[3], layer_sizes[-1])

    saes = keras.Sequential(name="saes")
    saes.add(keras.Input(shape=(layer_sizes[0],)))
    saes.add(layers.Dense(layer_sizes[1], name="hidden1"))
    saes.add(layers.Activation("sigmoid"))
    saes.add(layers.Dense(layer_sizes[2], name="hidden2"))
    saes.add(layers.Activation("sigmoid"))
    saes.add(layers.Dense(layer_sizes[3], name="hidden3"))
    saes.add(layers.Activation("sigmoid"))
    saes.add(layers.Dropout(0.2))
    saes.add(layers.Dense(layer_sizes[4], activation="sigmoid"))

    return [sae1, sae2, sae3, saes]
