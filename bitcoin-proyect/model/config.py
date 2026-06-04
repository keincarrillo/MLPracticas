# config.py
# fuente única de verdad para todas las constantes del proyecto
# si se quiere cambiar un hiperparámetro o ruta, solo se toca este archivo

# ── Datos ─────────────────────────────────────────────────────────────────────

TICKER      = "BTC-USD"
YEARS       = 5          # años de historial a descargar
WINDOW      = 60         # días de ventana deslizante para las secuencias LSTM
TRAIN_SPLIT = 0.8        # proporción de datos para entrenamiento (80/20)

# ── Rutas de artefactos ───────────────────────────────────────────────────────

MODEL_PATH     = "modelo.h5"
SCALER_PATH    = "scaler.pkl"
LOSS_DATA_PATH = "loss_data.json"

# ── Hiperparámetros del modelo ────────────────────────────────────────────────

SEED       = 42
UNITS      = 50    # neuronas en la capa LSTM
DROPOUT    = 0.2   # tasa de dropout para regularización
EPOCHS     = 20    # máximo de épocas (EarlyStopping puede detener antes)
BATCH_SIZE = 32
PATIENCE   = 5     # épocas sin mejora antes de detener el entrenamiento

# ── Forecast ─────────────────────────────────────────────────────────────────

FORECAST_DAYS = 30  # días a predecir hacia el futuro