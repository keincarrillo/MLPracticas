# main.py
# Pipeline principal del experimento MLP — Dataset genérico desde JSON.
#
# DISEÑO REFLEXIVO: el pipeline inspecciona el archivo JSON en tiempo de ejecución
# y se configura solo, sin que el usuario tenga que editar el código al cambiar
# de dataset. Sólo hay que cambiar las dos constantes de configuración al inicio:
#
#   DATASET_PATH  → ruta al archivo JSON
#   TARGET_COL    → nombre de la columna que contiene las etiquetas de clase
#
# Todo lo demás (número de features, clases, nombres, arquitectura de la red,
# nombres en gráficas) se detecta automáticamente.
#
# Estructura del pipeline:
#   [1] Carga reflexiva del JSON
#   [2] Definición automática de los 6 modelos según n_features y n_classes
#   [3] Entrenamiento, evaluación e impresión de resultados
#   [4] Tabla comparativa + 6 gráficas

import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.metrics         import classification_report
from collections             import Counter

from mlp_gradiente import MLPGradiente
from mlp_pso       import MLPPSO
from utilidades    import compute_metrics
from visualizaciones import (
    plot_learning_curves,
    plot_confusion_matrices,
    plot_metrics_comparison,
    plot_training_times,
    plot_pso_swarm,
    plot_weight_distributions,
    build_summary_table,
    set_class_names,
)

# ══════════════════════════════════════════════════════════════════════════════
#  ÚNICA SECCIÓN QUE CAMBIAR AL USAR UN DATASET DISTINTO
# ══════════════════════════════════════════════════════════════════════════════
DATASET_PATH = "iris.json"   # ruta al archivo JSON
TARGET_COL   = "especie"     # columna que contiene las etiquetas de clase
# ══════════════════════════════════════════════════════════════════════════════


# ── Utilidades de consola ─────────────────────────────────────────────────────

# print_banner(text, char, width) -> None
# Encabezado decorativo para separar visualmente las secciones del output.
def print_banner(text, char="═", width=70):
    pad = (width - len(text) - 2) // 2
    print(f"\n{char*width}")
    print(f"{char}{' '*pad}{text}{' '*(width-pad-len(text)-2)}{char}")
    print(f"{char*width}")


# ── Carga reflexiva del dataset ───────────────────────────────────────────────

# load_json_dataset(path, target_col) -> (X, y, feature_cols, label_encoder)
#
# Inspecciona el JSON en tiempo de ejecución para determinar:
#   - Qué columnas son features (todas las numéricas excepto target_col)
#   - Cuántas clases hay y cuáles son sus nombres
#
# Retorna:
#   X            : matriz (n_muestras, n_features) con valores numéricos
#   y            : vector de etiquetas enteras (0, 1, 2, ...)
#   feature_cols : lista con los nombres de las columnas de entrada
#   le           : LabelEncoder ajustado, para recuperar nombres de clases
#
# Por qué es reflexivo:
#   En lugar de hardcodear "sepal length (cm)", "sepal width (cm)", etc.,
#   detecta automáticamente cuáles columnas son numéricas. Si mañana usas
#   Wine, Penguins o tu propio CSV convertido a JSON, solo cambias DATASET_PATH
#   y TARGET_COL; el resto del código no se toca.
def load_json_dataset(path, target_col):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not raw:
        raise ValueError(f"El archivo '{path}' está vacío.")

    first = raw[0]

    # Detectar columnas de features: todas las que NO son target_col
    # y cuyos valores son numéricos (int o float).
    # Esto excluye automáticamente columnas de texto, IDs, fechas, etc.
    feature_cols = [
        col for col, val in first.items()
        if col != target_col and isinstance(val, (int, float))
    ]

    if not feature_cols:
        raise ValueError(
            f"No se encontraron columnas numéricas distintas de '{target_col}'.\n"
            f"Columnas disponibles: {list(first.keys())}"
        )

    # Verificar que todos los registros tienen las columnas esperadas
    # (detecta JSONs con filas incompletas antes de que falle numpy)
    missing = [
        i for i, r in enumerate(raw)
        if any(col not in r for col in feature_cols + [target_col])
    ]
    if missing:
        raise ValueError(
            f"Los registros {missing[:5]}{'...' if len(missing)>5 else ''} "
            f"tienen columnas faltantes."
        )

    # Construir X: extraer sólo las columnas numéricas de cada registro
    X = np.array([[r[col] for col in feature_cols] for r in raw], dtype=float)

    # Construir y: extraer la columna target y codificar como enteros
    # LabelEncoder asigna un entero por clase en orden alfabético.
    # Ej: setosa→0, versicolor→1, virginica→2
    raw_labels = [r[target_col] for r in raw]
    le = LabelEncoder()
    y  = le.fit_transform(raw_labels)

    return X, y, feature_cols, le


# ── Arquitectura automática ───────────────────────────────────────────────────

# build_configs(n_features, n_classes) -> list[(str, model)]
#
# Genera los 6 modelos adaptando la arquitectura al número de
# features y clases del dataset cargado.
#
# Heurística para tamaño de capas ocultas:
#   hidden_small = max(8, 2 * n_features)
#   hidden_large = max(16, 4 * n_features)
#
# Por qué esta heurística:
#   Una regla común es que la capa oculta tenga entre n_features y 2*n_features
#   neuronas. El max(8, ...) evita redes trivialmente pequeñas en datasets con
#   pocas features. Para datasets con muchas features (ej. 30 como Breast Cancer)
#   esto escala proporcionalmente sin que el usuario intervenga.
def build_configs(n_features, n_classes):
    h_small = max(8,  2 * n_features)
    h_large = max(16, 4 * n_features)

    arch_a = f"{n_features}→{h_small}→{n_classes}"
    arch_b = f"{n_features}→{h_large}→{h_small}→{n_classes}"

    return [
        # Batch GD sobre arquitectura pequeña: gradiente exacto, más estable
        (f"Gradient Clásico  [{arch_a}]",
         MLPGradiente(hidden_layers=(h_small,), lr=0.1, epochs=300, mode="classic")),

        # Mini-batch SGD sobre arquitectura pequeña: ruido → mejor generalización
        (f"Gradient Estocástico [{arch_a}]",
         MLPGradiente(hidden_layers=(h_small,), lr=0.05, epochs=300,
                      mode="stochastic", batch_size=16)),

        # Batch GD sobre red más profunda: más capacidad de aprender fronteras complejas
        (f"Gradient Clásico [{arch_b}]",
         MLPGradiente(hidden_layers=(h_large, h_small), lr=0.1, epochs=300,
                      mode="classic")),

        # SGD sobre red más profunda
        (f"Gradient Estocástico [{arch_b}]",
         MLPGradiente(hidden_layers=(h_large, h_small), lr=0.05, epochs=300,
                      mode="stochastic", batch_size=16)),

        # PSO clásico: exploración global sin gradientes
        (f"PSO Clásico [{arch_a}]",
         MLPPSO(hidden_layers=(h_small,), n_particles=30, epochs=200,
                w=0.7, c1=1.5, c2=1.5, mode="classic")),

        # PSO estocástico: fitness con mini-lotes → más rápido por generación
        (f"PSO Estocástico [{arch_a}]",
         MLPPSO(hidden_layers=(h_small,), n_particles=30, epochs=200,
                w=0.7, c1=1.5, c2=1.5, mode="stochastic", batch_size=32)),
    ]


# ── Evaluador central ─────────────────────────────────────────────────────────

# evaluate_model(model, X_train, y_train, X_test, y_test, name) -> dict
# Entrena el modelo y recopila todas las métricas de train y test.
# Retorna un diccionario estandarizado compatible con todas las funciones
# de visualización.
def evaluate_model(model, X_train, y_train, X_test, y_test, name):
    model.fit(X_train, y_train)

    y_pred_tr = model.predict(X_train)
    y_pred_te = model.predict(X_test)

    m_train = compute_metrics(y_train, y_pred_tr)
    m_test  = compute_metrics(y_test,  y_pred_te)

    result = {
        "name":             name,
        "model":            model,
        "train":            m_train,
        "test":             m_test,
        "train_time":       model.train_time,
        "epochs_run":       len(model.history),
        "converged_epoch":  model.converged_epoch,
        "y_test":           y_test,
        "y_pred_test":      y_pred_te,
        "history":          model.history,
        "accuracy_history": model.accuracy_history,
        "weight_history":   model.weight_history,
    }

    if isinstance(model, MLPPSO):
        result["pso"] = {
            "n_particles":          model.n_particles,
            "swarm_fitness_history":model.swarm_fitness_history,
            "diversity_history":    model.diversity_history,
            "personal_best_fitness":model.personal_best_fitness,
            "best_fitness":         min(model.swarm_fitness_history),
            "worst_fitness":        max(model.personal_best_fitness),
            "mean_pbest":           float(np.mean(model.personal_best_fitness)),
            "std_pbest":            float(np.std(model.personal_best_fitness)),
        }

    return result


# ── Impresión de resultados ───────────────────────────────────────────────────

# print_results(res, class_names) -> None
# Imprime métricas de un modelo. Recibe class_names desde el LabelEncoder
# para que el reporte use los nombres reales del dataset, no texto genérico.
def print_results(res, class_names):
    print_banner(res["name"])
    print(f"\n  {'Métrica':<20} {'Train':>12} {'Test':>12}")
    print(f"  {'-'*44}")
    for key in ["accuracy", "precision", "recall", "f1", "mcc"]:
        tr = res["train"].get(key, float("nan"))
        te = res["test"].get(key, float("nan"))
        print(f"  {key.capitalize():<20} {tr:>12.4f} {te:>12.4f}")

    print(f"\n  {'Tiempo entrenamiento':<30}: {res['train_time']*1000:.2f} ms")
    print(f"  {'Épocas ejecutadas':<30}: {res['epochs_run']}")
    conv = res["converged_epoch"] if res["converged_epoch"] else "No convergió"
    print(f"  {'Convergencia en época':<30}: {conv}")

    if "pso" in res:
        p = res["pso"]
        print(f"\n  Métricas PSO (enjambre):")
        print(f"    Partículas            : {p['n_particles']}")
        print(f"    Mejor fitness global  : {p['best_fitness']:.6f}")
        print(f"    Peor pbest            : {p['worst_fitness']:.6f}")
        print(f"    Media pbest           : {p['mean_pbest']:.6f} ± {p['std_pbest']:.6f}")

    print(f"\n  Reporte de clasificación (Test):")
    report = classification_report(
        res["y_test"], res["y_pred_test"], target_names=class_names
    )
    for line in report.splitlines():
        print("    " + line)


# ── Pipeline principal ────────────────────────────────────────────────────────

def main():
    print_banner("MLP MULTICAPA — CLASIFICACIÓN DESDE JSON", "█")

    # ── Paso 1: Carga reflexiva ───────────────────────────────────────────────
    print(f"\n[1/4] Cargando dataset desde '{DATASET_PATH}'...")
    print(f"       Columna objetivo: '{TARGET_COL}'")

    X, y, feature_cols, le = load_json_dataset(DATASET_PATH, TARGET_COL)

    # le.classes_ contiene los nombres originales de las clases en orden alfabético:
    # clases_[0]→entero 0, clases_[1]→entero 1, etc.
    class_names = list(le.classes_)
    n_features  = X.shape[1]
    n_classes   = len(class_names)

    # Pasar nombres de clases a visualizaciones para que las gráficas usen
    # los nombres reales (ej. "setosa") en lugar de texto genérico ("Clase 0")
    set_class_names(class_names)

    print(f"\n  Dataset detectado automáticamente:")
    print(f"  {'Muestras':<25}: {X.shape[0]}")
    print(f"  {'Features numéricas':<25}: {n_features}")
    print(f"  {'Columnas detectadas':<25}: {feature_cols}")
    print(f"  {f'Clases ({n_classes})':<25}: {class_names}")
    print(f"  {'Distribución':<25}: {dict(Counter(le.inverse_transform(y)))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Estandarización: fit solo en train para no filtrar información del test
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"\n  Train: {X_train.shape[0]} muestras | Test: {X_test.shape[0]} muestras")

    # ── Paso 2: Arquitectura automática ──────────────────────────────────────
    print(f"\n[2/4] Definiendo modelos (adaptados a {n_features} features, {n_classes} clases)...")
    configs = build_configs(n_features, n_classes)
    for name, _ in configs:
        print(f"  · {name}")

    # ── Paso 3: Entrenar y evaluar ────────────────────────────────────────────
    print("\n[3/4] Entrenando y evaluando modelos...\n")
    results = []
    for name, model in configs:
        print(f"  Entrenando: {name} ...")
        res = evaluate_model(model, X_train, y_train, X_test, y_test, name)
        results.append(res)
        print_results(res, class_names)

    # ── Paso 4: Tabla y visualizaciones ──────────────────────────────────────
    build_summary_table(results)

    print("\n[4/4] Generando visualizaciones...")
    plot_learning_curves(results)
    plot_confusion_matrices(results)
    plot_metrics_comparison(results)
    plot_training_times(results)
    plot_pso_swarm(results)
    plot_weight_distributions(results)

    print_banner("EJECUCIÓN COMPLETADA", "═")
    print("\n  Archivos generados:")
    for f in [
        "learning_curves.png", "confusion_matrices.png",
        "metrics_comparison.png", "training_times.png",
        "pso_swarm_analysis.png", "weight_distributions.png",
        "results_summary.csv",
    ]:
        print(f"    📄 {f}")
    print()


if __name__ == "__main__":
    main()
