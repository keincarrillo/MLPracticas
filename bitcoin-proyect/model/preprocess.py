# preprocess.py
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

TICKER = "BTC-USD"
YEARS = 3
WINDOW = 60
TRAIN_SPLIT = 0.8
SCALER_PATH = "scaler.pkl"


def download_data() -> pd.DataFrame:
    """Descarga los últimos YEARS años de precio de cierre de Bitcoin."""
    df = yf.download(TICKER, period=f"{YEARS}y", interval="1d", auto_adjust=True)
    df = df[["Close"]].dropna()
    return df


def normalize(data: np.ndarray) -> tuple[np.ndarray, MinMaxScaler]:
    """Normaliza al rango [0, 1] y guarda el scaler en disco."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    return scaled, scaler


def build_sequences(scaled: np.ndarray, window: int = WINDOW):
    """Construye pares (X, y) con ventana deslizante."""
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window:i, 0])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y)


def split_and_reshape(X: np.ndarray, y: np.ndarray, split: float = TRAIN_SPLIT):
    """Divide en train/test y hace reshape a (N, window, 1) para la LSTM."""
    cut = int(len(X) * split)
    X_train, X_test = X[:cut], X[cut:]
    y_train, y_test = y[:cut], y[cut:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test  = X_test.reshape((X_test.shape[0],  X_test.shape[1],  1))

    return X_train, X_test, y_train, y_test


def load_scaler() -> MinMaxScaler:
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    print("Descargando datos...")
    df = download_data()
    print(f"  {len(df)} registros descargados ({df.index[0].date()} → {df.index[-1].date()})")

    prices = df["Close"].values.reshape(-1, 1)
    scaled, _ = normalize(prices)
    print(f"  Scaler guardado en {SCALER_PATH}")

    X, y = build_sequences(scaled)
    X_train, X_test, y_train, y_test = split_and_reshape(X, y)

    print(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape} | y_test: {y_test.shape}")
    print("Preprocesamiento completado.")
