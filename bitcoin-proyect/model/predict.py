# predict.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import json as _json
tf.get_logger().setLevel("ERROR")

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error

from preprocess import download_data, build_sequences, split_and_reshape, load_scaler

MODEL_PATH    = "modelo.h5"
WINDOW        = 60
FORECAST_DAYS = 30


# get_predictions() -> tuple[list[float], list[float]]
# carga el modelo entrenado y genera predicciones sobre el conjunto de prueba
# usa load_scaler() para no sobreescribir el scaler guardado durante el entrenamiento
# retorna (precios_reales, precios_predichos) en escala original USD
def get_predictions():
    df     = download_data()
    prices = df["Close"].values.reshape(-1, 1)
    scaler = load_scaler()
    scaled = scaler.transform(prices)

    X, y   = build_sequences(scaled)
    X_train, X_test, y_train, y_test = split_and_reshape(X, y)

    model = load_model(MODEL_PATH)

    pred_scaled = model.predict(X_test, verbose=0)
    pred        = scaler.inverse_transform(pred_scaled)
    real        = scaler.inverse_transform(y_test.reshape(-1, 1))

    return real.flatten().tolist(), pred.flatten().tolist()


# get_metrics() -> float
# calcula el error absoluto medio (MAE) sobre el conjunto de prueba
# retorna el MAE redondeado a 2 decimales en USD
def get_metrics():
    real, pred = get_predictions()
    mae        = mean_absolute_error(real, pred)
    return round(mae, 2)


# get_history() -> tuple[list[str], list[float]]
# retorna el historial completo de precios de cierre con sus fechas formateadas
def get_history():
    df     = download_data()
    prices = df["Close"].values.flatten().tolist()
    dates  = df.index.strftime("%Y-%m-%d").tolist()
    return dates, prices


# forecast_next_days(days: int) -> tuple[list[str], list[float]]
# genera prediccion recursiva de los proximos days dias
# usa load_scaler() para no sobreescribir el scaler guardado durante el entrenamiento
# usa el ultimo window de precios conocidos como semilla y extiende paso a paso
def forecast_next_days(days: int = FORECAST_DAYS):
    df     = download_data()
    prices = df["Close"].values.reshape(-1, 1)
    scaler = load_scaler()
    scaled = scaler.transform(prices)
    model  = load_model(MODEL_PATH)

    # semilla: ultimos WINDOW valores escalados
    ventana = list(scaled[-WINDOW:].flatten())
    futuro  = []

    for _ in range(days):
        x    = np.array(ventana[-WINDOW:]).reshape(1, WINDOW, 1)
        pred = model.predict(x, verbose=0)[0][0]
        futuro.append(pred)
        ventana.append(pred)

    precios_futuros = scaler.inverse_transform(
        np.array(futuro).reshape(-1, 1)
    ).flatten().tolist()

    ultima_fecha  = df.index[-1]
    fechas_futuro = pd.date_range(start=ultima_fecha + pd.Timedelta(days=1), periods=days)
    fechas_futuro = fechas_futuro.strftime("%Y-%m-%d").tolist()

    return fechas_futuro, precios_futuros


# plot_history() -> None
# genera y guarda la grafica del precio historico completo de Bitcoin
# corresponde a la primera de las tres graficas requeridas por el proyecto
def plot_history():
    dates, prices = get_history()
    plt.figure(figsize=(12, 5))
    plt.plot(prices, label="Precio de cierre", color="steelblue")
    plt.title("Precio historico de Bitcoin (BTC-USD)")
    plt.xlabel("Dias")
    plt.ylabel("Precio USD")
    plt.legend()
    plt.tight_layout()
    plt.savefig("history.png")
    print("  Grafica guardada en history.png")


# plot_predictions() -> None
# genera y guarda la grafica de precio real vs predicho sobre el conjunto de prueba
# corresponde a la tercera de las tres graficas requeridas por el proyecto
def plot_predictions():
    real, pred = get_predictions()
    plt.figure(figsize=(12, 5))
    plt.plot(real, label="Real",     color="steelblue")
    plt.plot(pred, label="Predicho", color="tomato", linestyle="--")
    plt.title("Precio real vs predicho — conjunto de prueba")
    plt.xlabel("Dias")
    plt.ylabel("Precio USD")
    plt.legend()
    plt.tight_layout()
    plt.savefig("predictions.png")
    print("  Grafica guardada en predictions.png")


# -----------------------------------------------------------------------
# modo API — flags para ser llamado desde la API REST en Bun
# --history  imprime JSON con fechas y precios historicos
# --predict  imprime JSON con precios reales y predichos del conjunto de prueba
# --metrics  imprime JSON con el MAE calculado
# -----------------------------------------------------------------------

def run_api_mode(flag: str) -> None:
    if flag == "--history":
        dates, prices = get_history()
        print(json.dumps({"dates": dates, "prices": prices}))

    elif flag == "--predict":
        real, pred = get_predictions()
        print(json.dumps({"real": real, "predicted": pred}))

    elif flag == "--metrics":
        mae = get_metrics()
        print(json.dumps({"mae": mae}))

    elif flag == "--forecast":
        fechas, precios = forecast_next_days()
        print(json.dumps({"dates": fechas, "prices": precios}))
    
    elif flag == "--loss":

        with open("loss_data.json", "r") as f:
            data = _json.load(f)
        epochs = list(range(1, len(data["train"]) + 1))
        print(_json.dumps({"epochs": epochs, "train": data["train"], "val": data["val"]}))

    else:
        print(json.dumps({"error": f"flag desconocido: {flag}"}), file=sys.stderr)
        sys.exit(1)


# -----------------------------------------------------------------------
# punto de entrada — genera las tres graficas, MAE y forecast
# si se pasa un flag de API solo imprime JSON y termina
# -----------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_api_mode(sys.argv[1])
    else:
        print("Generando grafica de historial...")
        plot_history()

        print("Generando predicciones...")
        plot_predictions()

        mae = get_metrics()
        print(f"  MAE: ${mae:,.2f} USD")

        print("Generando forecast 30 dias...")
        fechas, precios = forecast_next_days()
        for fecha, precio in zip(fechas, precios):
            print(f"  {fecha}: ${precio:,.2f}")