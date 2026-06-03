# tests/test_preprocess.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler
from preprocess import build_sequences, split_and_reshape


# PT-01 — normalizacion en rango valido
# verifica que todos los valores escalados queden en [0, 1]
def test_normalizacion_rango():
    prices = np.array([100, 200, 150, 300, 250]).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)
    assert scaled.min() >= 0.0
    assert scaled.max() <= 1.0


# PT-02 — numero de secuencias generadas
# con N datos y ventana W se deben generar N - W secuencias
def test_numero_secuencias():
    prices = np.arange(10).reshape(-1, 1).astype(float)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)
    X, y   = build_sequences(scaled, window=3)
    assert len(X) == 7


# PT-03 — longitud de cada secuencia
# cada muestra X[i] debe tener exactamente window elementos
def test_longitud_secuencia():
    prices = np.arange(20).reshape(-1, 1).astype(float)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)
    window = 5
    X, y   = build_sequences(scaled, window=window)
    for seq in X:
        assert len(seq) == window


# PT-04 — reshape tridimensional no pierde muestras
# el total de muestras train + test debe coincidir con el total original
# ademas la tercera dimension del reshape debe ser 1 (requerido por LSTM)
def test_reshape_tridimensional():
    prices = np.arange(100).reshape(-1, 1).astype(float)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)
    X, y   = build_sequences(scaled, window=10)
    X_train, X_test, y_train, y_test = split_and_reshape(X, y)
    total_original = len(X)
    total_reshape  = X_train.shape[0] + X_test.shape[0]
    assert total_original  == total_reshape
    assert X_train.shape[2] == 1
    assert X_test.shape[2]  == 1


# PT-05 — desnormalizacion recupera valores originales
# aplicar inverse_transform sobre datos escalados debe retornar los precios originales
def test_desnormalizacion():
    prices   = np.array([1000, 2000, 3000, 4000, 5000]).reshape(-1, 1).astype(float)
    scaler   = MinMaxScaler()
    scaled   = scaler.fit_transform(prices)
    recovery = scaler.inverse_transform(scaled)
    diff     = np.abs(recovery - prices)
    assert diff.max() < 0.01