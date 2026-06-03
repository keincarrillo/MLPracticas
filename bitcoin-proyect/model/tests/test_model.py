# tests/test_model.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
import tensorflow as tf
from train import build_model


# PT-06 — numero de capas correcto
# la arquitectura debe tener exactamente 3 capas: LSTM, Dropout y Dense
def test_numero_capas():
    model = build_model(window=60)
    assert len(model.layers) == 3


# PT-07 — forma de salida correcta
# dado un batch de 1 muestra la salida debe ser de forma (1, 1)
def test_forma_salida():
    model  = build_model(window=60)
    x      = np.random.rand(1, 60, 1).astype(np.float32)
    output = model.predict(x, verbose=0)
    assert output.shape == (1, 1)


# PT-08 — prediccion produce valor valido
# la prediccion no debe ser NaN ni infinita para entrada aleatoria valida
def test_prediccion_valida():
    model  = build_model(window=60)
    x      = np.random.rand(1, 60, 1).astype(np.float32)
    output = model.predict(x, verbose=0)
    assert not np.isnan(output[0][0])
    assert not np.isinf(output[0][0])


# PT-09 — funcion de perdida correcta
# el modelo debe compilarse con mean_squared_error como funcion de perdida
def test_funcion_perdida():
    model = build_model(window=60)
    assert model.loss == "mean_squared_error"