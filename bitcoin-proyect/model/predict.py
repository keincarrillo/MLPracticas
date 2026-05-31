# predict.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error
import json

from preprocess import download_data, normalize, build_sequences, split_and_reshape, load_scaler

MODEL_PATH = "modelo.h5"
WINDOW = 60
FORECAST_DAYS = 30


def get_predictions():
    """Carga el modelo y genera predicciones sobre el conjunto de prueba."""
    df = download_data()
    prices = df["Close"].values.reshape(-1, 1)
    scaled, _ = normalize(prices)
    X, y = build_sequences(scaled)
    X_train, X_test, y_train, y_test = split_and_reshape(X, y)

    model = load_model(MODEL_PATH)
    scaler = load_scaler()

    pred_scaled = model.predict(X_test)
    pred = scaler.inverse_transform(pred_scaled)
    real = scaler.inverse_transform(y_test.reshape(-1, 1))

    return real.flatten().tolist(), pred.flatten().tolist()


def get_metrics():
    """Calcula el MAE sobre el conjunto de prueba."""
    real, pred = get_predictions()
    mae = mean_absolute_error(real, pred)
    return round(mae, 2)


def get_history():
    """Devuelve el historial completo de precios como lista."""
    df = download_data()
    prices = df["Close"].values.flatten().tolist()
    dates = df.index.strftime("%Y-%m-%d").tolist()
    return dates, prices


def forecast_next_days(days: int = FORECAST_DAYS):
    """Genera predicción de los próximos DAYS días."""
    df = download_data()
    prices = df["Close"].values.reshape(-1, 1)
    scaled, _ = normalize(prices)
    scaler = load_scaler()
    model = load_model(MODEL_PATH)

    window = list(scaled[-WINDOW:].flatten())
    future = []
    for _ in range(days):
        x = np.array(window[-WINDOW:]).reshape(1, WINDOW, 1)
        pred = model.predict(x, verbose=0)[0][0]
        future.append(pred)
        window.append(pred)

    future_prices = scaler.inverse_transform(
        np.array(future).reshape(-1, 1)
    ).flatten().tolist()

    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
    future_dates = future_dates.strftime("%Y-%m-%d").tolist()

    return future_dates, future_prices


def plot_predictions():
    """Genera y guarda la gráfica de real vs predicho."""
    real, pred = get_predictions()
    plt.figure(figsize=(12, 5))
    plt.plot(real, label="Real", color="steelblue")
    plt.plot(pred, label="Predicho", color="tomato", linestyle="--")
    plt.title("Precio real vs predicho — conjunto de prueba")
    plt.xlabel("Días")
    plt.ylabel("Precio USD")
    plt.legend()
    plt.tight_layout()
    plt.savefig("predictions.png")
    print("  Gráfica guardada en predictions.png")


if __name__ == "__main__":
    print("Generando predicciones...")
    plot_predictions()

    mae = get_metrics()
    print(f"  MAE: ${mae:,.2f} USD")

    print("Generando forecast 30 días...")
    dates, prices = forecast_next_days()
    for d, p in zip(dates, prices):
        print(f"  {d}: ${p:,.2f}")