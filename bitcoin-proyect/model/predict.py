# predict.py
# generación de predicciones, métricas y forecast a partir del modelo entrenado
#
# problema original: cada función llamaba download_data() por separado,
# causando hasta 3 descargas de Yahoo Finance por request concurrente
#
# solución: _load_pipeline() centraliza la carga de datos, scaler y modelo
# en una sola llamada y cachea el resultado en módulo-level para que
# invocaciones sucesivas dentro del mismo proceso no vuelvan a descargar

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from config import MODEL_PATH, LOSS_DATA_PATH, WINDOW, FORECAST_DAYS
from preprocess import download_data, load_scaler, build_sequences, split_and_reshape


# ── Caché de pipeline ─────────────────────────────────────────────────────────

# _cache almacena el resultado de _load_pipeline() para evitar descargas
# repetidas dentro del mismo proceso; None indica que aún no se ha cargado
_cache: dict | None = None


# _load_pipeline() -> dict
# descarga datos, carga scaler y modelo una única vez por proceso
# las llamadas posteriores devuelven el resultado cacheado sin I/O adicional
# retorna un dict con: df, prices, scaled, model, scaler, X_test, y_test
def _load_pipeline() -> dict:
    global _cache
    if _cache is not None:
        return _cache

    df     = download_data()
    prices = df["Close"].values.reshape(-1, 1)
    scaler = load_scaler()
    scaled = scaler.transform(prices)

    X, y = build_sequences(scaled)
    _, X_test, _, y_test = split_and_reshape(X, y)

    model = load_model(MODEL_PATH)

    _cache = {
        "df":     df,
        "prices": prices,
        "scaled": scaled,
        "scaler": scaler,
        "model":  model,
        "X_test": X_test,
        "y_test": y_test,
    }
    return _cache


# ── Predicciones sobre el conjunto de prueba ──────────────────────────────────

# get_predictions() -> tuple[list[float], list[float]]
# genera predicciones sobre el conjunto de prueba usando el pipeline cacheado
# retorna (precios_reales, precios_predichos) en escala original USD
def get_predictions() -> tuple[list[float], list[float]]:
    p = _load_pipeline()

    pred_scaled = p["model"].predict(p["X_test"], verbose=0)
    pred        = p["scaler"].inverse_transform(pred_scaled)
    real        = p["scaler"].inverse_transform(p["y_test"].reshape(-1, 1))

    return real.flatten().tolist(), pred.flatten().tolist()


# ── Métricas ──────────────────────────────────────────────────────────────────

# get_metrics() -> floaft
# calcula el error absoluto medio (MAE) sobre el conjunto de prueba
# retorna el MAE redondeado a 2 decimales en USD
def get_metrics() -> float:
    real, pred = get_predictions()
    return round(mean_absolute_error(real, pred), 2)


# ── Historial ─────────────────────────────────────────────────────────────────

# get_history() -> tuple[list[str], list[float]]
# retorna el historial completo de precios de cierre con fechas formateadas
def get_history() -> tuple[list[str], list[float]]:
    p      = _load_pipeline()
    prices = p["prices"].flatten().tolist()
    dates  = p["df"].index.strftime("%Y-%m-%d").tolist()
    return dates, prices


# ── Forecast ──────────────────────────────────────────────────────────────────

# forecast_next_days(days) -> tuple[list[str], list[float]]
# genera predicción recursiva de los próximos days días
# usa el último WINDOW de precios conocidos como semilla y extiende paso a paso
def forecast_next_days(days: int = FORECAST_DAYS) -> tuple[list[str], list[float]]:
    p      = _load_pipeline()
    model  = p["model"]
    scaler = p["scaler"]
    scaled = p["scaled"]
    df     = p["df"]

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
    fechas_futuro = pd.date_range(
        start=ultima_fecha + pd.Timedelta(days=1), periods=days
    ).strftime("%Y-%m-%d").tolist()

    return fechas_futuro, precios_futuros


# ── Visualizaciones (modo standalone) ────────────────────────────────────────

# plot_history() -> None
# genera y guarda la gráfica del precio histórico completo de Bitcoin
def plot_history() -> None:
    dates, prices = get_history()
    plt.figure(figsize=(12, 5))
    plt.plot(prices, label="Precio de cierre", color="steelblue")
    plt.title("Precio histórico de Bitcoin (BTC-USD)")
    plt.xlabel("Días")
    plt.ylabel("Precio USD")
    plt.legend()
    plt.tight_layout()
    plt.savefig("history.png")
    print("  Gráfica guardada en 'history.png'")


# plot_predictions() -> None
# genera y guarda la gráfica de precio real vs predicho sobre el conjunto de prueba
def plot_predictions() -> None:
    real, pred = get_predictions()
    plt.figure(figsize=(12, 5))
    plt.plot(real, label="Real",     color="steelblue")
    plt.plot(pred, label="Predicho", color="tomato", linestyle="--")
    plt.title("Precio real vs predicho — conjunto de prueba")
    plt.xlabel("Días")
    plt.ylabel("Precio USD")
    plt.legend()
    plt.tight_layout()
    plt.savefig("predictions.png")
    print("  Gráfica guardada en 'predictions.png'")


# ── Dispatcher CLI (modo API) ─────────────────────────────────────────────────

# _HANDLERS mapea cada flag a su función de dominio
# agregar un nuevo endpoint = añadir una entrada aquí, sin tocar el if/elif
_HANDLERS: dict[str, callable] = {
    "--history": lambda: (lambda d, p: {"dates": d, "prices": p})(*get_history()),
    "--predict": lambda: (lambda r, p: {"real": r, "predicted": p})(*get_predictions()),
    "--metrics": lambda: {"mae": get_metrics()},
    "--forecast": lambda: (lambda d, p: {"dates": d, "prices": p})(*forecast_next_days()),
    "--loss": lambda: (
        lambda data: {
            "epochs": list(range(1, len(data["train"]) + 1)),
            "train":  data["train"],
            "val":    data["val"],
        }
    )(json.load(open(LOSS_DATA_PATH))),
}


# run_api_mode(flag) -> None
# ejecuta el handler correspondiente al flag y emite JSON por stdout
# usado exclusivamente cuando el script es invocado desde la API de Bun
def run_api_mode(flag: str) -> None:
    handler = _HANDLERS.get(flag)
    if handler is None:
        print(json.dumps({"error": f"flag desconocido: {flag}"}), file=sys.stderr)
        sys.exit(1)

    result = handler()
    print(json.dumps(result))


# ── Punto de entrada ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_api_mode(sys.argv[1])
    else:
        print("Generando gráfica de historial...")
        plot_history()

        print("Generando predicciones...")
        plot_predictions()

        mae = get_metrics()
        print(f"  MAE: ${mae:,.2f} USD")

        print("Generando forecast 30 días...")
        fechas, precios = forecast_next_days()
        for fecha, precio in zip(fechas, precios):
            print(f"  {fecha}: ${precio:,.2f}")