# main.py
# Pipeline principal del experimento.
# Carga el dataset, preprocesa, entrena los 6 modelos y genera todas las métricas.
#
# Estructura del experimento:
#   [1] Carga y partición del dataset Breast Cancer Wisconsin
#   [2] Definición de los 6 modelos con sus hiperparámetros
#   [3] Entrenamiento, evaluación e impresión de resultados por modelo
#   [4] Tabla comparativa resumen + 7 gráficas guardadas como PNG
#
# Para ejecutar:
#   python main.py
#
# Para usar en Google Colab, pegar este archivo junto con los demás módulos
# o montarlos en Drive. Alternativamente, copiar todo en una celda.

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Importar los tres modelos implementados
from perceptron_delta     import PerceptronDelta
from perceptron_gradiente import PerceptronGradient
from perceptron_pso       import PerceptronPSO

# Importar utilidades y visualizaciones
from utilidades      import compute_metrics
from visualizaciones import (
    plot_learning_curves,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_metrics_comparison,
    plot_training_times,
    plot_pso_swarm,
    plot_weight_distributions,
    build_summary_table,
)

# ── Impresión de banners ──────────────────────────────────────────────────────

# print_banner(text: str, char: str, width: int) -> None
# Imprime un encabezado decorativo centrado entre líneas de caracteres.
# Útil para separar visualmente las secciones del output en consola.
def print_banner(text, char="═", width=70):
    pad = (width - len(text) - 2) // 2
    print(f"\n{char*width}")
    print(f"{char}{' '*pad}{text}{' '*(width-pad-len(text)-2)}{char}")
    print(f"{char*width}")

# ── Evaluador central ─────────────────────────────────────────────────────────

# evaluate_model(model, X_train, y_train, X_test, y_test, name) -> dict
# Entrena un modelo y calcula todas las métricas disponibles.
# Retorna un diccionario con:
#   - métricas de train y test (accuracy, precision, recall, f1, mcc, auc_roc)
#   - historial de entrenamiento (pérdida, exactitud, norma de pesos por época)
#   - estadísticas de los pesos finales
#   - métricas de enjambre si el modelo es PSO
def evaluate_model(model, X_train, y_train, X_test, y_test, name):
    model.fit(X_train, y_train)

    y_pred_tr = model.predict(X_train)
    y_pred_te = model.predict(X_test)

    # Obtener probabilidades si el modelo las soporta (todos los implementados las soportan)
    has_proba = hasattr(model, "predict_proba")
    y_prob_tr = model.predict_proba(X_train) if has_proba else None
    y_prob_te = model.predict_proba(X_test)  if has_proba else None

    m_train = compute_metrics(y_train, y_pred_tr, y_prob_tr)
    m_test  = compute_metrics(y_test,  y_pred_te, y_prob_te)

    result = {
        "name":            name,
        "model":           model,
        "train":           m_train,
        "test":            m_test,
        "train_time":      model.train_time,
        "epochs_run":      len(model.history),
        "converged_epoch": model.converged_epoch,
        "y_test":          y_test,
        "y_pred_test":     y_pred_te,
        "y_prob_test":     y_prob_te,
        "history":         model.history,
        "accuracy_history":model.accuracy_history,
        "weight_history":  model.weight_history,
        # Estadísticas descriptivas de los pesos finales del modelo
        "weight_stats": {
            "mean": float(np.mean(model.weights)),
            "std":  float(np.std(model.weights)),
            "min":  float(np.min(model.weights)),
            "max":  float(np.max(model.weights)),
            "norm": float(np.linalg.norm(model.weights)),
        },
    }

    # Para PSO agregar métricas de enjambre al diccionario de resultados
    if isinstance(model, PerceptronPSO):
        result["pso"] = {
            "n_particles":          model.n_particles,
            "swarm_fitness_history":model.swarm_fitness_history,
            "diversity_history":    model.diversity_history,
            "personal_best_fitness":model.personal_best_fitness,
            # Mejor fitness alcanzado por el enjambre en toda la ejecución
            "best_fitness":         min(model.swarm_fitness_history),
            # Peor fitness personal: indica qué tan mal quedó la partícula más rezagada
            "worst_fitness":        max(model.personal_best_fitness),
            "mean_pbest":           float(np.mean(model.personal_best_fitness)),
            "std_pbest":            float(np.std(model.personal_best_fitness)),
        }

    return result

# ── Impresión de resultados de un modelo ─────────────────────────────────────

# print_results(res: dict) -> None
# Imprime en consola todas las métricas de un modelo de forma tabular y legible.
# Incluye: tabla de métricas train/test, tiempo, convergencia,
# estadísticas de pesos, métricas PSO (si aplica) y reporte de clasificación.
def print_results(res):
    print_banner(res["name"])
    print(f"\n  {'Métrica':<20} {'Train':>12} {'Test':>12}")
    print(f"  {'-'*44}")
    for key in ["accuracy", "precision", "recall", "f1", "mcc", "auc_roc"]:
        tr = res["train"].get(key, float("nan"))
        te = res["test"].get(key, float("nan"))
        label = key.upper() if key == "auc_roc" else key.capitalize()
        print(f"  {label:<20} {tr:>12.4f} {te:>12.4f}")

    print(f"\n  {'Tiempo entrenamiento':<30}: {res['train_time']*1000:.2f} ms")
    print(f"  {'Épocas ejecutadas':<30}: {res['epochs_run']}")
    conv = res["converged_epoch"] if res["converged_epoch"] else "No convergió"
    print(f"  {'Convergencia en época':<30}: {conv}")

    ws = res["weight_stats"]
    print(f"\n  Estadísticas de pesos finales:")
    print(f"    Media={ws['mean']:.4f}  Std={ws['std']:.4f}  "
          f"Min={ws['min']:.4f}  Max={ws['max']:.4f}  Norm={ws['norm']:.4f}")

    # Sección exclusiva para modelos PSO
    if "pso" in res:
        p = res["pso"]
        print(f"\n  Métricas PSO (enjambre):")
        print(f"    Partículas            : {p['n_particles']}")
        print(f"    Mejor fitness global  : {p['best_fitness']:.6f}")
        print(f"    Peor pbest            : {p['worst_fitness']:.6f}")
        print(f"    Media pbest           : {p['mean_pbest']:.6f} ± {p['std_pbest']:.6f}")

    print(f"\n  Reporte de clasificación (Test):")
    report = classification_report(res["y_test"], res["y_pred_test"],
                                   target_names=["Maligno", "Benigno"])
    for line in report.splitlines():
        print("    " + line)

# ── Pipeline principal ────────────────────────────────────────────────────────

def main():
    print_banner("PERCEPTRÓN MULTICONFIGURACION — BREAST CANCER WISCONSIN", "█")

    # ── Paso 1: Cargar y preparar el dataset ─────────────────────────────────
    print("\n[1/4] Cargando dataset Breast Cancer Wisconsin...")
    data = load_breast_cancer()
    X = data.data.astype(float)   # 569 muestras, 30 características numéricas
    y = data.target.astype(float) # 0=maligno, 1=benigno

    print(f"  Muestras:        {X.shape[0]}")
    print(f"  Características: {X.shape[1]}")
    print(f"  Clases: {dict(zip(data.target_names, np.bincount(y.astype(int))))}")

    # Dividir en 80% entrenamiento / 20% prueba conservando proporción de clases
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Estandarizar: media=0, desviación=1 por característica.
    # Crítico para gradiente descendente y PSO; sin esto los pesos divergen.
    # IMPORTANTE: fit solo en train para no contaminar el test con información del futuro.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"  Train: {X_train.shape[0]} muestras | Test: {X_test.shape[0]} muestras")

    # ── Paso 2: Definir los 6 modelos ────────────────────────────────────────
    print("\n[2/4] Definiendo modelos...")

    # Cada tupla es (nombre_descriptivo, instancia_del_modelo)
    configs = [
        # Perceptrón clásico de Rosenblatt: activación dura, regla Delta, batch
        ("Escalón+Delta Clásico",
         PerceptronDelta(lr=0.01, epochs=200, mode="classic")),

        # Igual que el anterior pero actualiza pesos ejemplo a ejemplo (online)
        ("Escalón+Delta Estocástico",
         PerceptronDelta(lr=0.01, epochs=200, mode="stochastic")),

        # Activación suave, gradiente exacto sobre todo el batch (logística batch GD)
        ("Sigmoide+Gradient Clásico",
         PerceptronGradient(lr=0.1, epochs=300, mode="classic")),

        # Igual pero con SGD; más ruidoso pero puede converger más rápido en práctica
        ("Sigmoide+Gradient Estocástico",
         PerceptronGradient(lr=0.05, epochs=300, mode="stochastic")),

        # PSO: 40 partículas exploran el espacio de pesos evaluando el dataset completo
        ("Sigmoide+PSO Clásico",
         PerceptronPSO(n_particles=40, epochs=200, w=0.7, c1=1.5, c2=1.5,
                       mode="classic")),

        # PSO estocástico: cada generación evalúa solo 64 ejemplos para mayor velocidad
        ("Sigmoide+PSO Estocástico",
         PerceptronPSO(n_particles=30, epochs=200, w=0.7, c1=1.5, c2=1.5,
                       mode="stochastic", batch_size=64)),
    ]

    # ── Paso 3: Entrenar y evaluar ───────────────────────────────────────────
    print("\n[3/4] Entrenando y evaluando modelos...\n")
    results = []
    for name, model in configs:
        print(f"  Entrenando: {name} ...")
        res = evaluate_model(model, X_train, y_train, X_test, y_test, name)
        results.append(res)
        print_results(res)

    # ── Paso 4: Tabla resumen y visualizaciones ──────────────────────────────
    build_summary_table(results)

    print("\n[4/4] Generando visualizaciones...")
    plot_learning_curves(results)      # pérdida, exactitud y norma por época
    plot_confusion_matrices(results)   # matrices de confusión
    plot_roc_curves(results)           # curvas ROC con AUC
    plot_metrics_comparison(results)   # barras comparativas de métricas
    plot_training_times(results)       # tiempos de entrenamiento
    plot_pso_swarm(results)            # análisis del enjambre PSO
    plot_weight_distributions(results) # histogramas de pesos finales

    print_banner("EJECUCIÓN COMPLETADA", "═")
    print("\n  Archivos generados:")
    for f in [
        "learning_curves.png", "confusion_matrices.png", "roc_curves.png",
        "metrics_comparison.png", "training_times.png", "pso_swarm_analysis.png",
        "weight_distributions.png", "results_summary.csv",
    ]:
        print(f"    📄 {f}")
    print()


if __name__ == "__main__":
    main()