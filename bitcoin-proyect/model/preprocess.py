# preprocess.py
# pipeline de descarga, normalización y construcción de secuencias
# cada función tiene una sola responsabilidad; ninguna mezcla I/O con transformación

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import pickle

from config import TICKER, YEARS, WINDOW, TRAIN_SPLIT, SCALER_PATH


# ── Descarga ──────────────────────────────────────────────────────────────────

# download_data() -> pd.DataFrame
# descarga los últimos YEARS años de precio de cierre de Bitcoin desde Yahoo Finance
# elimina filas con valores nulos y retorna solo la columna Close
# lanza RuntimeError si la descarga falla o retorna un DataFrame vacío
def download_data() -> pd.DataFrame:
    try:
        df = yf.download(TICKER, period=f"{YEARS}y", interval="1d", auto_adjust=True)
    except Exception as exc:
        raise RuntimeError(f"Error al descargar datos de {TICKER}: {exc}") from exc

    df = df[["Close"]].dropna()

    if df.empty:
        raise RuntimeError(
            f"Yahoo Finance devolvió un DataFrame vacío para {TICKER}. "
            "Verifica la conexión o el ticker."
        )

    return df


# ── Normalización ─────────────────────────────────────────────────────────────

# fit_and_save_scaler(data) -> tuple[np.ndarray, MinMaxScaler]
# ajusta un MinMaxScaler sobre los datos y lo guarda en disco
# separado de transform() para dejar claro que este es el único punto
# donde el scaler se "aprende" — solo debe llamarse desde train.py
def fit_and_save_scaler(data: np.ndarray) -> tuple[np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    return scaled, scaler


# load_scaler() -> MinMaxScaler
# carga el scaler previamente ajustado y guardado en disco
# usar esta función en predict.py garantiza que no se sobreescriba el scaler
# lanza FileNotFoundError si el scaler no existe todavía
def load_scaler() -> MinMaxScaler:
    try:
        with open(SCALER_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Scaler no encontrado en '{SCALER_PATH}'. "
            "Ejecuta preprocess.py antes de predecir."
        )


# ── Secuencias ────────────────────────────────────────────────────────────────

# build_sequences(scaled, window) -> tuple[np.ndarray, np.ndarray]
# construye pares (X, y) con ventana deslizante de tamaño window
# X[i] contiene los últimos window precios normalizados, y[i] es el precio siguiente
def build_sequences(
    scaled: np.ndarray, window: int = WINDOW
) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window : i, 0])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y)


# split_and_reshape(X, y, split) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
# divide en conjuntos train/test según la proporción split (80/20 por defecto)
# hace reshape a (N, window, 1) requerido por la capa LSTM
def split_and_reshape(
    X: np.ndarray, y: np.ndarray, split: float = TRAIN_SPLIT
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    corte   = int(len(X) * split)
    X_train = X[:corte].reshape(-1, X.shape[1], 1)
    X_test  = X[corte:].reshape(-1, X.shape[1], 1)
    y_train = y[:corte]
    y_test  = y[corte:]
    return X_train, X_test, y_train, y_test


# ── Punto de entrada ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Descargando datos...")
    df = download_data()
    print(f"  {len(df)} registros ({df.index[0].date()} → {df.index[-1].date()})")

    prices, scaler = fit_and_save_scaler(df["Close"].values.reshape(-1, 1))
    print(f"  Scaler guardado en '{SCALER_PATH}'")

    X, y = build_sequences(prices)
    X_train, X_test, y_train, y_test = split_and_reshape(X, y)

    print(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape} | y_test: {y_test.shape}")
    print("Preprocesamiento completado.")