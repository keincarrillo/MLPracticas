# utilidades.py
# Funciones auxiliares compartidas por todos los módulos de perceptrón.
# Contiene: activaciones, métricas, instalación de dependencias.

# ── Instalación silenciosa de dependencias (útil en Colab) ──────────────────
import subprocess, sys

# _install(pkg: str) -> None
# Instala un paquete pip si no está disponible en el entorno actual.
# Se usa al inicio para garantizar que sklearn, matplotlib, etc. existan.
def _install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

try:
    import sklearn
except ImportError:
    _install("scikit-learn")

try:
    import matplotlib
except ImportError:
    _install("matplotlib")

try:
    import seaborn
except ImportError:
    _install("seaborn")

try:
    import pandas
except ImportError:
    _install("pandas")

# ── Importaciones estándar ───────────────────────────────────────────────────
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef
)

# ── Reproducibilidad ─────────────────────────────────────────────────────────

# set_seed(seed: int) -> None
# Fija la semilla de numpy para que los experimentos sean reproducibles.
# Llamar antes de inicializar pesos o mezclar datos.
def set_seed(seed=42):
    np.random.seed(seed)

# ── Funciones de activación ──────────────────────────────────────────────────

# sigmoid(x: np.ndarray) -> np.ndarray
# Función sigmoide: transforma cualquier valor real al rango (0, 1).
# Se usa como función de activación suave y para calcular probabilidades.
# El clip evita overflow en exponentes muy grandes o muy negativos.
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

# step(x: np.ndarray) -> np.ndarray
# Función escalón de Heaviside: retorna 1.0 si x >= 0, 0.0 si x < 0.
# Es la activación clásica del perceptrón original de Rosenblatt.
# No es diferenciable, por eso no se puede usar con gradiente descendente.
def step(x):
    return (x >= 0.0).astype(float)

# ── Métricas de clasificación ────────────────────────────────────────────────

# compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
#                 y_prob: np.ndarray | None) -> dict
# Calcula un conjunto completo de métricas de clasificación binaria.
# Si se proporcionan probabilidades (y_prob), también calcula AUC-ROC.
# Retorna un diccionario con claves: accuracy, precision, recall, f1, mcc, auc_roc.
def compute_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        # Fracción de predicciones correctas sobre el total
        "accuracy":  accuracy_score(y_true, y_pred),
        # De todos los que predije como positivos, ¿cuántos lo eran?
        "precision": precision_score(y_true, y_pred, zero_division=0),
        # De todos los positivos reales, ¿cuántos detecté?
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        # Media armónica de precision y recall; equilibra ambas
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        # Coeficiente de correlación de Matthews: robusto con clases desbalanceadas
        "mcc":       matthews_corrcoef(y_true, y_pred),
    }
    if y_prob is not None:
        # Área bajo la curva ROC: 1.0 es perfecto, 0.5 es azar
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["auc_roc"] = float("nan")
    return metrics