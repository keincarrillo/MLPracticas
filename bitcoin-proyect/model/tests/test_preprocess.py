# tests/test_preprocess.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler
from preprocess import build_sequences, split_and_reshape


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def scaled_prices():
    """Array de precios normalizados reutilizable entre tests."""
    prices = np.linspace(100, 500, 100).reshape(-1, 1)
    scaler = MinMaxScaler()
    return scaler.fit_transform(prices), scaler

@pytest.fixture
def sequences(scaled_prices):
    """Par (X, y) construido con ventana de 10 para tests de split."""
    scaled, _ = scaled_prices
    return build_sequences(scaled, window=10)


# ── PT-01 — normalización en rango válido ─────────────────────────────────────
# verifica que todos los valores escalados queden estrictamente en [0, 1]

def test_normalizacion_rango():
    prices = np.array([100, 200, 150, 300, 250]).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)
    assert scaled.min() >= 0.0
    assert scaled.max() <= 1.0


# ── PT-02 — número de secuencias generadas ────────────────────────────────────
# con N datos y ventana W se deben generar exactamente N - W secuencias

def test_numero_secuencias(scaled_prices):
    scaled, _ = scaled_prices
    X, y = build_sequences(scaled, window=3)
    assert len(X) == len(scaled) - 3


# ── PT-03 — longitud de cada secuencia ───────────────────────────────────────
# cada muestra X[i] debe tener exactamente window elementos

def test_longitud_secuencia(scaled_prices):
    scaled, _ = scaled_prices
    window    = 5
    X, _      = build_sequences(scaled, window=window)
    assert all(len(seq) == window for seq in X)


# ── PT-04 — reshape tridimensional no pierde muestras ────────────────────────
# total train + test debe coincidir con el total original
# la tercera dimensión del reshape debe ser 1 (requerido por LSTM)

def test_reshape_tridimensional(sequences):
    X, y = sequences
    X_train, X_test, y_train, y_test = split_and_reshape(X, y)
    assert X_train.shape[0] + X_test.shape[0] == len(X)
    assert X_train.shape[2] == 1
    assert X_test.shape[2]  == 1


# ── PT-05 — desnormalización recupera valores originales ─────────────────────
# aplicar inverse_transform sobre datos escalados debe retornar los precios originales

def test_desnormalizacion():
    prices   = np.array([1000, 2000, 3000, 4000, 5000]).reshape(-1, 1).astype(float)
    scaler   = MinMaxScaler()
    scaled   = scaler.fit_transform(prices)
    recovery = scaler.inverse_transform(scaled)
    assert np.abs(recovery - prices).max() < 0.01


# ── PT-10 — split respeta la proporción ──────────────────────────────────────
# con split=0.8 el conjunto de entrenamiento debe ser ≥ 79% del total

def test_split_proporcion(sequences):
    X, y = sequences
    X_train, X_test, _, _ = split_and_reshape(X, y, split=0.8)
    ratio = X_train.shape[0] / (X_train.shape[0] + X_test.shape[0])
    assert 0.79 <= ratio <= 0.81


# ── PT-11 — build_sequences alinea X con y ───────────────────────────────────
# y[i] debe ser el elemento que sigue inmediatamente a X[i] en el array original

def test_secuencias_alineadas(scaled_prices):
    scaled, _ = scaled_prices
    window    = 5
    X, y      = build_sequences(scaled, window=window)
    # el último elemento de X[0] es scaled[window-1]; y[0] debe ser scaled[window]
    assert X[0][-1] == pytest.approx(scaled[window - 1, 0])
    assert y[0]     == pytest.approx(scaled[window, 0])