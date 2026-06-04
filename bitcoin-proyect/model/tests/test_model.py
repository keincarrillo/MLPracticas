# tests/test_model.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
import tensorflow as tf
from train import build_model


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def model():
    """Modelo con ventana de 60 días reutilizable entre tests."""
    return build_model(window=60)


# ── PT-06 — número de capas correcto ─────────────────────────────────────────
# la arquitectura debe tener exactamente 3 capas: LSTM, Dropout y Dense

def test_numero_capas(model):
    assert len(model.layers) == 3


# ── PT-07 — forma de salida correcta ─────────────────────────────────────────
# dado un batch de 1 muestra la salida debe ser de forma (1, 1)

def test_forma_salida(model):
    x      = np.random.rand(1, 60, 1).astype(np.float32)
    output = model.predict(x, verbose=0)
    assert output.shape == (1, 1)


# ── PT-08 — predicción produce valor válido ───────────────────────────────────
# la predicción no debe ser NaN ni infinita para entrada aleatoria válida

def test_prediccion_valida(model):
    x      = np.random.rand(1, 60, 1).astype(np.float32)
    output = model.predict(x, verbose=0)
    assert not np.isnan(output[0][0])
    assert not np.isinf(output[0][0])


# ── PT-09 — función de pérdida correcta ──────────────────────────────────────
# el modelo debe compilarse con mean_squared_error como función de pérdida

def test_funcion_perdida(model):
    assert model.loss == "mean_squared_error"


# ── PT-12 — el modelo responde igual ante la misma entrada (determinismo) ─────
# con la misma semilla aleatoria dos predicciones deben ser idénticas

def test_prediccion_determinista():
    x      = np.random.rand(1, 60, 1).astype(np.float32)
    model1 = build_model(window=60)
    model2 = build_model(window=60)
    out1   = model1.predict(x, verbose=0)
    out2   = model2.predict(x, verbose=0)
    # dos modelos recién inicializados con la misma semilla deben dar igual
    np.testing.assert_array_almost_equal(out1, out2, decimal=6)


# ── PT-13 — la capa LSTM tiene las unidades correctas ─────────────────────────
# la primera capa debe ser LSTM con UNITS neuronas

def test_unidades_lstm(model):
    from config import UNITS
    lstm_layer = model.layers[0]
    assert isinstance(lstm_layer, tf.keras.layers.LSTM)
    assert lstm_layer.units == UNITS


# ── PT-14 — la tasa de Dropout es la configurada ─────────────────────────────

def test_tasa_dropout(model):
    from config import DROPOUT
    dropout_layer = model.layers[1]
    assert isinstance(dropout_layer, tf.keras.layers.Dropout)
    assert dropout_layer.rate == pytest.approx(DROPOUT)