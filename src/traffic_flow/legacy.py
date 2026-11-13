"""
Load the original Keras 2.1 ``.h5`` files (``artifacts/legacy``) into Keras 3 models.

``keras.saving.load_model`` cannot read those files, but the raw weight arrays inside
are in the standard layout, so we rebuild the architecture (``legacy=True``) and copy the
arrays over layer by layer.
"""
import json

import h5py
import numpy as np

from traffic_flow import models


def _read_layer_weights(path):
    """Return ``[(layer_name, [arrays...]), ...]`` for layers that have weights."""
    out = []
    with h5py.File(path, "r") as f:
        group = f["model_weights"]
        for name in [n.decode() if isinstance(n, bytes) else n
                     for n in group.attrs["layer_names"]]:
            g = group[name]
            weight_names = [n.decode() if isinstance(n, bytes) else n
                            for n in g.attrs["weight_names"]]
            if weight_names:
                out.append((name, [np.array(g[w]) for w in weight_names]))
    return out


def load_legacy_weights(model, path):
    """Copy weights from a Keras 2.1 ``.h5`` file into ``model`` (same architecture)."""
    saved = _read_layer_weights(path)
    targets = [layer for layer in model.layers if layer.weights]
    if len(saved) != len(targets):
        raise ValueError("%s has %d weighted layers, model has %d"
                         % (path, len(saved), len(targets)))
    for (name, arrays), layer in zip(saved, targets):
        shapes = [tuple(w.shape) for w in layer.get_weights()]
        if shapes != [tuple(a.shape) for a in arrays]:
            raise ValueError("shape mismatch in %s -> %s: %s vs %s"
                             % (name, layer.name, [a.shape for a in arrays], shapes))
        layer.set_weights(arrays)
    return model


def load_legacy_model(name, path, lags=12):
    """Build the legacy-flavoured architecture for ``name`` and load ``path`` into it."""
    if name == "lstm":
        model = models.get_lstm([lags, 64, 64, 1], legacy=True)
    elif name == "gru":
        model = models.get_gru([lags, 64, 64, 1], legacy=True)
    elif name == "saes":
        model = models.get_saes([lags, 400, 400, 400, 1])[-1]
    else:
        raise ValueError("unknown model: %s" % name)
    return load_legacy_weights(model, path)
