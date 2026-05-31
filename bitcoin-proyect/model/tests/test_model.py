# tests/test_model.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
import tensorflow as tf
from train import build_model


# PT-06 — Número de capas correcto
def test_numero_capas():
    model = build_model(window=60)
    assert len(model.layers) == 3


# PT-07 — Forma de salida correcta
def test_forma_salida():
    model = build_model(window=60)
    x = np.random.rand(1, 60, 1).astype(np.float32)
    output = model.predict(x, verbose=0)
    assert output.shape == (1, 1)


# PT-08 — Predicción produce valor válido
def test_prediccion_valida():
    model = build_model(window=60)
    x = np.random.rand(1, 60, 1).astype(np.float32)
    output = model.predict(x, verbose=0)
    assert not np.isnan(output[0][0])
    assert not np.isinf(output[0][0])


# PT-09 — Función de pérdida correcta
def test_funcion_perdida():
    model = build_model(window=60)
    assert model.loss == "mean_squared_error"