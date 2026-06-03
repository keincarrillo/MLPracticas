# preprocess.py
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import pickle

TICKER      = "BTC-USD"
YEARS       = 3
WINDOW      = 60
TRAIN_SPLIT = 0.8
SCALER_PATH = "scaler.pkl"


# download_data() -> pd.DataFrame
# descarga los ultimos YEARS años de precio de cierre de Bitcoin desde Yahoo Finance
# elimina filas con valores nulos y retorna solo la columna Close
def download_data() -> pd.DataFrame:
    df = yf.download(TICKER, period=f"{YEARS}y", interval="1d", auto_adjust=True)
    df = df[["Close"]].dropna()
    return df


# normalize(data: np.ndarray) -> tuple[np.ndarray, MinMaxScaler]
# normaliza los precios al rango [0, 1] usando MinMaxScaler
# guarda el scaler en disco para poder invertir la transformacion en predict.py
# nota: llamar esta funcion sobreescribe el scaler guardado — solo debe llamarse en train.py
def normalize(data: np.ndarray) -> tuple[np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    return scaled, scaler


# build_sequences(scaled: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]
# construye pares (X, y) con ventana deslizante de tamaño window
# X[i] contiene los ultimos window precios normalizados, y[i] es el precio siguiente
def build_sequences(scaled: np.ndarray, window: int = WINDOW):
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window:i, 0])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y)


# split_and_reshape(X, y, split) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
# divide en conjuntos train/test segun la proporcion split (80/20 por defecto)
# hace reshape a (N, window, 1) requerido por la capa LSTM
def split_and_reshape(X: np.ndarray, y: np.ndarray, split: float = TRAIN_SPLIT):
    corte   = int(len(X) * split)
    X_train = X[:corte]
    X_test  = X[corte:]
    y_train = y[:corte]
    y_test  = y[corte:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test  = X_test.reshape((X_test.shape[0],   X_test.shape[1],  1))

    return X_train, X_test, y_train, y_test


# load_scaler() -> MinMaxScaler
# carga el scaler previamente guardado en disco por normalize()
# usar esta funcion en predict.py garantiza que no se sobreescriba el scaler del entrenamiento
def load_scaler() -> MinMaxScaler:
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)


# -----------------------------------------------------------------------
# punto de entrada — verificacion del pipeline de preprocesamiento
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("Descargando datos...")
    df = download_data()
    print(f"  {len(df)} registros descargados ({df.index[0].date()} -> {df.index[-1].date()})")

    prices    = df["Close"].values.reshape(-1, 1)
    scaled, _ = normalize(prices)
    print(f"  Scaler guardado en {SCALER_PATH}")

    X, y = build_sequences(scaled)
    X_train, X_test, y_train, y_test = split_and_reshape(X, y)

    print(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape} | y_test: {y_test.shape}")
    print("Preprocesamiento completado.")