# train.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

from preprocess import download_data, normalize, build_sequences, split_and_reshape

SEED = 42
UNITS = 50
DROPOUT = 0.2
EPOCHS = 20
BATCH_SIZE = 32
MODEL_PATH = "modelo.h5"


def build_model(window: int = 60) -> Sequential:
    tf.random.set_seed(SEED)
    model = Sequential([
        LSTM(UNITS, input_shape=(window, 1), return_sequences=False),
        Dropout(DROPOUT),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def plot_loss(history):
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="Entrenamiento")
    plt.plot(history.history["val_loss"], label="Validación")
    plt.title("Curva de aprendizaje")
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    print("  Gráfica guardada en loss_curve.png")


if __name__ == "__main__":
    print("Cargando y preprocesando datos...")
    df = download_data()
    prices = df["Close"].values.reshape(-1, 1)
    scaled, _ = normalize(prices)
    X, y = build_sequences(scaled)
    X_train, X_test, y_train, y_test = split_and_reshape(X, y)

    print("Construyendo modelo...")
    model = build_model()
    model.summary()

    print("\nEntrenando...")
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    model.save(MODEL_PATH)
    print(f"\n  Modelo guardado en {MODEL_PATH}")

    plot_loss(history)
    print("Entrenamiento completado.")