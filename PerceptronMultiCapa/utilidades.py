# utilidades.py
# Funciones auxiliares compartidas por todos los módulos del MLP.
# Contiene: activaciones, métricas multiclase, reproducibilidad.
#
# Diferencias respecto a la práctica anterior (perceptrón binario):
#   - Se agrega softmax para clasificación multiclase (Iris tiene 3 clases)
#   - compute_metrics ahora usa macro-averaging y soporta más de 2 clases
#   - Se mantiene sigmoid para las capas ocultas del MLP

import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Instalación silenciosa de dependencias ────────────────────────────────────
import subprocess, sys

def _install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg, name in [("sklearn", "scikit-learn"), ("matplotlib", "matplotlib"),
                  ("seaborn", "seaborn"), ("pandas", "pandas")]:
    try:
        __import__(pkg)
    except ImportError:
        _install(name)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

# ── Reproducibilidad ──────────────────────────────────────────────────────────

# set_seed(seed: int) -> None
# Fija la semilla de numpy para garantizar que los experimentos sean reproducibles.
# Llamar siempre antes de inicializar pesos o mezclar datos.
def set_seed(seed=42):
    np.random.seed(seed)

# ── Funciones de activación ───────────────────────────────────────────────────

# sigmoid(x: np.ndarray) -> np.ndarray
# Transforma cualquier valor real al rango (0, 1).
# Se usa en las CAPAS OCULTAS del MLP para introducir no-linealidad.
# Sin ella, apilar capas sería equivalente a tener una sola capa lineal.
# El clip evita overflow numérico en exponentes muy grandes.
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

# sigmoid_derivative(x: np.ndarray) -> np.ndarray
# Derivada de la función sigmoide evaluada en x (ya pasado por sigmoid).
# Se necesita en el backpropagation para calcular el gradiente de la pérdida
# respecto a las activaciones de las capas ocultas.
# Fórmula: σ'(x) = σ(x) * (1 - σ(x))
def sigmoid_derivative(s):
    # s ya es sigmoid(x), no x crudo
    return s * (1.0 - s)

# softmax(x: np.ndarray) -> np.ndarray
# Convierte un vector de puntuaciones (logits) en una distribución de probabilidad.
# La suma de las salidas siempre es 1, lo que permite interpretar cada salida
# como la probabilidad de pertenecer a cada clase.
# Se usa SOLO en la capa de salida para clasificación multiclase.
# El truco de restar el máximo (x - max) evita overflow numérico sin cambiar el resultado.
def softmax(x):
    # x puede ser (n_muestras, n_clases) o (n_clases,)
    if x.ndim == 1:
        e = np.exp(x - np.max(x))
        return e / e.sum()
    else:
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

# relu(x: np.ndarray) -> np.ndarray
# Rectified Linear Unit: max(0, x).
# Alternativa a sigmoid para capas ocultas; no satura en valores positivos.
# No se usa en este proyecto pero se incluye como referencia.
def relu(x):
    return np.maximum(0.0, x)

# ── Función de pérdida ────────────────────────────────────────────────────────

# cross_entropy(y_true_oh: np.ndarray, y_prob: np.ndarray) -> float
# Calcula la entropía cruzada categórica para clasificación multiclase.
# y_true_oh: etiquetas en formato one-hot, forma (n_muestras, n_clases)
# y_prob:    probabilidades predichas por softmax, forma (n_muestras, n_clases)
# Retorna un escalar: el promedio de la pérdida sobre todas las muestras.
# La pérdida es 0 cuando las predicciones son perfectas; crece cuando son malas.
def cross_entropy(y_true_oh, y_prob):
    eps = 1e-15
    p = np.clip(y_prob, eps, 1 - eps)
    # Solo la probabilidad de la clase correcta contribuye a la pérdida
    return -np.mean(np.sum(y_true_oh * np.log(p), axis=1))

# ── Codificación one-hot ──────────────────────────────────────────────────────

# to_one_hot(y: np.ndarray, n_classes: int) -> np.ndarray
# Convierte etiquetas enteras [0, 1, 2] a vectores one-hot.
# Ejemplo: clase 1 con 3 clases → [0, 1, 0]
# Se necesita porque la capa de salida softmax produce un vector de probabilidades
# y la pérdida de entropía cruzada requiere comparar vectores, no escalares.
def to_one_hot(y, n_classes):
    oh = np.zeros((len(y), n_classes))
    oh[np.arange(len(y)), y.astype(int)] = 1.0
    return oh

# ── Métricas de clasificación ─────────────────────────────────────────────────

# compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict
# Calcula métricas de clasificación multiclase con macro-averaging.
# Macro-averaging trata todas las clases por igual, independientemente de su tamaño.
# Es la opción correcta para Iris, que tiene clases balanceadas.
# Retorna diccionario con: accuracy, precision, recall, f1, mcc.
def compute_metrics(y_true, y_pred):
    return {
        # Fracción de predicciones correctas sobre el total de muestras
        "accuracy":  accuracy_score(y_true, y_pred),
        # Promedio de la precisión por clase: de lo que predije como clase k, ¿cuánto era clase k?
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        # Promedio del recall por clase: de todos los de clase k, ¿cuántos detecté?
        "recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        # Media armónica de precision y recall, promediada entre clases
        "f1":        f1_score(y_true, y_pred, average="macro", zero_division=0),
        # MCC multiclase: robusto ante desbalance; 1=perfecto, 0=azar, -1=inverso
        "mcc":       matthews_corrcoef(y_true, y_pred),
    }
