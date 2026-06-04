# train.py
# construcción y entrenamiento del modelo LSTM
# cada función tiene una sola responsabilidad

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

from config import (
    SEED, UNITS, DROPOUT, EPOCHS, BATCH_SIZE, PATIENCE,
    MODEL_PATH, LOSS_DATA_PATH,
)
from preprocess import download_data, fit_and_save_scaler, build_sequences, split_and_reshape


# ── Arquitectura ──────────────────────────────────────────────────────────────

# build_model(window) -> Sequential
# construye la arquitectura de la red neuronal LSTM
# LSTM(UNITS) → Dropout(DROPOUT) → Dense(1)
# compilado con Adam y MSE como función de pérdida
def build_model(window: int = 60) -> Sequential:
    tf.random.set_seed(SEED)
    model = Sequential([
        LSTM(UNITS, input_shape=(window, 1), return_sequences=False),
        Dropout(DROPOUT),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# ── Persistencia de métricas ──────────────────────────────────────────────────

# save_loss_data(history) -> None
# guarda los valores de loss por época en un archivo JSON para la API REST
# separado de plot_loss() para respetar SRP — persistencia ≠ visualización
def save_loss_data(history: tf.keras.callbacks.History) -> None:
    loss_data = {
        "train": history.history["loss"],
        "val":   history.history["val_loss"],
    }
    with open(LOSS_DATA_PATH, "w") as f:
        json.dump(loss_data, f)
    print(f"  Datos de pérdida guardados en '{LOSS_DATA_PATH}'")


# plot_loss(history) -> None
# genera y guarda la gráfica de curva de aprendizaje (PNG)
# separado de save_loss_data() — visualización ≠ persistencia
def plot_loss(history: tf.keras.callbacks.History) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"],     label="Entrenamiento")
    plt.plot(history.history["val_loss"], label="Validación")
    plt.title("Curva de aprendizaje")
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    print("  Gráfica guardada en 'loss_curve.png'")


# ── Punto de entrada ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Cargando y preprocesando datos...")
    df               = download_data()
    prices           = df["Close"].values.reshape(-1, 1)
    scaled, _        = fit_and_save_scaler(prices)
    X, y             = build_sequences(scaled)
    X_train, X_test, y_train, y_test = split_and_reshape(X, y)

    print("Construyendo modelo...")
    model = build_model(window=X_train.shape[1])
    model.summary()

    print("\nEntrenando...")
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
    )
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
    )

    model.save(MODEL_PATH)
    print(f"\n  Modelo guardado en '{MODEL_PATH}'")

    save_loss_data(history)
    plot_loss(history)
    print("Entrenamiento completado.")