# visualizaciones.py
# Módulo de gráficas para comparar y analizar los resultados de los perceptrones.
# Genera 7 figuras y las guarda como PNG en el directorio de trabajo.
#
# Funciones disponibles:
#   plot_learning_curves     → pérdida, exactitud y norma de pesos por época
#   plot_confusion_matrices  → matrices de confusión para cada modelo
#   plot_roc_curves          → curvas ROC con AUC para todos los modelos
#   plot_metrics_comparison  → barras comparativas de todas las métricas
#   plot_training_times      → tiempos de entrenamiento en milisegundos
#   plot_pso_swarm           → métricas específicas del enjambre PSO
#   plot_weight_distributions → histograma de pesos finales por modelo
#   build_summary_table      → tabla resumen y guardado en CSV

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, classification_report

# Paleta de colores para distinguir los 6 modelos en las gráficas
COLORS = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261", "#264653"]

# _setup_style() -> None
# Configura el tema oscuro de matplotlib para todas las gráficas.
# Se llama al inicio de cada función de graficado para mantener consistencia visual.
def _setup_style():
    plt.rcParams.update({
        "figure.facecolor": "#0d1117",  # fondo de la figura (gris muy oscuro)
        "axes.facecolor":   "#161b22",  # fondo del área de graficado
        "axes.edgecolor":   "#30363d",  # color del borde de los ejes
        "axes.labelcolor":  "#c9d1d9",  # color de las etiquetas de ejes
        "xtick.color":      "#c9d1d9",  # color de las marcas del eje X
        "ytick.color":      "#c9d1d9",  # color de las marcas del eje Y
        "text.color":       "#c9d1d9",  # color del texto general
        "grid.color":       "#21262d",  # color de la cuadrícula
        "grid.linewidth":   0.5,
        "legend.facecolor": "#161b22",
        "legend.edgecolor": "#30363d",
        "font.family":      "monospace",
    })

# plot_learning_curves(results_list: list[dict]) -> None
# Genera una figura con 3 subgráficas en una fila:
#   izquierda : pérdida (BCE o MSE) por época para cada modelo
#   centro    : exactitud por época
#   derecha   : norma L2 de los pesos por época (mide si los pesos crecen o se estabilizan)
# Guarda la figura en "learning_curves.png"
def plot_learning_curves(results_list):
    _setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Curvas de Aprendizaje — Pérdida, Exactitud y Norma de Pesos",
                 fontsize=13, color="#c9d1d9", y=1.02)

    for i, res in enumerate(results_list):
        col = COLORS[i % len(COLORS)]
        ep = range(1, len(res["history"]) + 1)
        axes[0].plot(ep, res["history"], color=col, lw=1.5, label=res["name"])
        axes[1].plot(ep, res["accuracy_history"], color=col, lw=1.5)
        axes[2].plot(ep, res["weight_history"], color=col, lw=1.5)

    for ax, title in zip(axes, ["Pérdida (BCE / MSE)", "Exactitud", "Norma de pesos"]):
        ax.set_xlabel("Época")
        ax.set_title(title, color="#c9d1d9")
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.show()
    print("  → Guardado: learning_curves.png")

# plot_confusion_matrices(results_list: list[dict]) -> None
# Genera una cuadrícula de matrices de confusión, una por modelo.
# Cada celda muestra el conteo de: VP (arriba-derecha), FN, FP, VN.
# Los colores más intensos indican conteos mayores.
# Guarda la figura en "confusion_matrices.png"
def plot_confusion_matrices(results_list):
    _setup_style()
    n = len(results_list)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5))
    axes = np.array(axes).flatten()
    fig.suptitle("Matrices de Confusión (Conjunto de Prueba)", fontsize=13, color="#c9d1d9")

    for i, res in enumerate(results_list):
        cm = confusion_matrix(res["y_test"], res["y_pred_test"])
        col = COLORS[i % len(COLORS)]
        cmap = sns.light_palette(col, as_cmap=True)
        sns.heatmap(cm, ax=axes[i], annot=True, fmt="d", cmap=cmap,
                    linewidths=0.5, linecolor="#21262d",
                    annot_kws={"size": 14, "color": "#0d1117"},
                    cbar=False)
        axes[i].set_title(res["name"], fontsize=9, color="#c9d1d9")
        axes[i].set_xlabel("Predicho", fontsize=8)
        axes[i].set_ylabel("Real", fontsize=8)
        axes[i].set_xticklabels(["Maligno", "Benigno"], fontsize=8)
        axes[i].set_yticklabels(["Maligno", "Benigno"], fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.show()
    print("  → Guardado: confusion_matrices.png")

# plot_roc_curves(results_list: list[dict]) -> None
# Grafica las curvas ROC de todos los modelos en un mismo eje.
# La línea diagonal punteada representa un clasificador aleatorio (AUC=0.5).
# Cuanto más cercana al vértice superior izquierdo, mejor es la curva.
# Guarda la figura en "roc_curves.png"
def plot_roc_curves(results_list):
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "--", color="#555", lw=0.8, label="Azar")

    for i, res in enumerate(results_list):
        if res["y_prob_test"] is None:
            continue
        col = COLORS[i % len(COLORS)]
        fpr, tpr, _ = roc_curve(res["y_test"], res["y_prob_test"])
        auc = res["test"]["auc_roc"]
        ax.plot(fpr, tpr, color=col, lw=1.8, label=f"{res['name']}  (AUC={auc:.3f})")

    ax.set_xlabel("Tasa de Falsos Positivos")
    ax.set_ylabel("Tasa de Verdaderos Positivos")
    ax.set_title("Curvas ROC", color="#c9d1d9")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.show()
    print("  → Guardado: roc_curves.png")

# plot_metrics_comparison(results_list: list[dict]) -> None
# Genera un gráfico de barras agrupadas con las 6 métricas clave en el conjunto de prueba.
# Permite comparar visualmente todos los modelos en cada métrica de un vistazo.
# Guarda la figura en "metrics_comparison.png"
def plot_metrics_comparison(results_list):
    _setup_style()
    metrics = ["accuracy", "precision", "recall", "f1", "mcc", "auc_roc"]
    labels  = ["Accuracy", "Precision", "Recall", "F1", "MCC", "AUC-ROC"]
    x = np.arange(len(metrics))
    width = 0.8 / len(results_list)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, res in enumerate(results_list):
        vals = [res["test"].get(m, 0) for m in metrics]
        offset = (i - len(results_list) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width=width * 0.9,
                      color=COLORS[i % len(COLORS)], label=res["name"],
                      alpha=0.85, edgecolor="#0d1117", linewidth=0.5)
        # Anotar el valor numérico encima de cada barra
        for bar, v in zip(bars, vals):
            if not (v != v):  # verificar que no es NaN
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=5.5, color="#c9d1d9")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_title("Comparación de Métricas en Test", fontsize=13, color="#c9d1d9")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("metrics_comparison.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.show()
    print("  → Guardado: metrics_comparison.png")

# plot_training_times(results_list: list[dict]) -> None
# Gráfica de barras horizontales con el tiempo de entrenamiento de cada modelo.
# Útil para comparar el costo computacional de cada estrategia de aprendizaje.
# Guarda la figura en "training_times.png"
def plot_training_times(results_list):
    _setup_style()
    names  = [r["name"] for r in results_list]
    times  = [r["train_time"] * 1000 for r in results_list]  # convertir a milisegundos
    colors = [COLORS[i % len(COLORS)] for i in range(len(results_list))]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, times, color=colors, edgecolor="#0d1117", linewidth=0.5, alpha=0.85)
    for bar, t in zip(bars, times):
        ax.text(t + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{t:.1f} ms", va="center", fontsize=9, color="#c9d1d9")

    ax.set_xlabel("Tiempo (ms)")
    ax.set_title("Tiempo de Entrenamiento por Modelo", color="#c9d1d9")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_times.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.show()
    print("  → Guardado: training_times.png")

# plot_pso_swarm(results_list: list[dict]) -> None
# Genera visualizaciones específicas del enjambre para los modelos PSO.
# Por cada modelo PSO produce 3 subgráficas:
#   izquierda : evolución del fitness global (gbest) por generación
#   centro    : diversidad del enjambre (cuánto explodan las partículas)
#   derecha   : distribución del fitness personal (pbest) al final
# Guarda la figura en "pso_swarm_analysis.png"
def plot_pso_swarm(results_list):
    # Filtrar solo los modelos que tienen métricas PSO
    pso_results = [r for r in results_list if "pso" in r]
    if not pso_results:
        return

    _setup_style()
    n = len(pso_results)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4.5 * n))
    # Garantizar que axes siempre sea una lista de filas, incluso con 1 modelo PSO
    if n == 1:
        axes = [axes]
    fig.suptitle("Análisis del Enjambre PSO", fontsize=13, color="#c9d1d9")

    for row, res in enumerate(pso_results):
        p = res["pso"]
        col = COLORS[results_list.index(res) % len(COLORS)]
        epochs = range(1, len(p["swarm_fitness_history"]) + 1)

        # Gráfica 1: fitness del mejor global disminuye a medida que el enjambre converge
        axes[row][0].plot(epochs, p["swarm_fitness_history"], color=col, lw=1.5)
        axes[row][0].set_title(f"{res['name']} — Fitness gbest", fontsize=9)
        axes[row][0].set_xlabel("Generación")
        axes[row][0].grid(True, alpha=0.3)

        # Gráfica 2: diversidad; si cae rápido el enjambre puede haberse estancado prematuramente
        axes[row][1].plot(epochs, res["model"].diversity_history, color="#f4a261", lw=1.5)
        axes[row][1].set_title(f"{res['name']} — Diversidad del enjambre", fontsize=9)
        axes[row][1].set_xlabel("Generación")
        axes[row][1].grid(True, alpha=0.3)

        # Gráfica 3: histograma del fitness personal; idealmente debería estar concentrado
        # cerca del fitness global (pocas partículas rezagadas)
        axes[row][2].hist(p["personal_best_fitness"], bins=12,
                          color=col, edgecolor="#0d1117", alpha=0.85)
        axes[row][2].axvline(p["best_fitness"], color="#fff",
                             linestyle="--", lw=1.2, label=f"Mejor={p['best_fitness']:.4f}")
        axes[row][2].axvline(p["mean_pbest"], color="#e9c46a",
                             linestyle="--", lw=1.2, label=f"Media={p['mean_pbest']:.4f}")
        axes[row][2].set_title(f"{res['name']} — Distribución pbest", fontsize=9)
        axes[row][2].legend(fontsize=7)
        axes[row][2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pso_swarm_analysis.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.show()
    print("  → Guardado: pso_swarm_analysis.png")

# plot_weight_distributions(results_list: list[dict]) -> None
# Histograma de los pesos finales de cada modelo.
# Una distribución concentrada cerca de 0 indica pesos pequeños (regularizados).
# Una distribución amplia indica que el modelo usa fuertemente varias características.
# Guarda la figura en "weight_distributions.png"
def plot_weight_distributions(results_list):
    _setup_style()
    n = len(results_list)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5))
    axes = np.array(axes).flatten()
    fig.suptitle("Distribución de Pesos Finales", fontsize=13, color="#c9d1d9")

    for i, res in enumerate(results_list):
        col = COLORS[i % len(COLORS)]
        w = res["model"].weights
        axes[i].hist(w, bins=20, color=col, edgecolor="#0d1117", alpha=0.85, density=True)
        axes[i].axvline(np.mean(w), color="#fff", lw=1.2, linestyle="--",
                        label=f"μ={np.mean(w):.3f}")
        axes[i].set_title(res["name"], fontsize=9, color="#c9d1d9")
        axes[i].legend(fontsize=7)
        axes[i].grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig("weight_distributions.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.show()
    print("  → Guardado: weight_distributions.png")

# build_summary_table(results_list: list[dict]) -> pd.DataFrame
# Construye y muestra en consola una tabla comparativa con todas las métricas clave.
# También guarda la tabla en "results_summary.csv" para análisis externo.
# Retorna el DataFrame para uso programático si se desea.
def build_summary_table(results_list):
    rows = []
    for res in results_list:
        row = {
            "Modelo":       res["name"],
            "Acc Train":    f"{res['train']['accuracy']:.4f}",
            "Acc Test":     f"{res['test']['accuracy']:.4f}",
            "F1 Test":      f"{res['test']['f1']:.4f}",
            "AUC Test":     f"{res['test']['auc_roc']:.4f}" if res['test']['auc_roc'] == res['test']['auc_roc'] else "N/A",
            "MCC Test":     f"{res['test']['mcc']:.4f}",
            "Tiempo(ms)":   f"{res['train_time'] * 1000:.1f}",
            "Convergencia": str(res["converged_epoch"]) if res["converged_epoch"] else "—",
        }
        # Añadir columnas exclusivas de PSO si el modelo las tiene
        if "pso" in res:
            row["PSO Partículas"] = str(res["pso"]["n_particles"])
            row["PSO Mejor Fit"]  = f"{res['pso']['best_fitness']:.5f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n" + "═" * 80)
    print("  TABLA RESUMEN COMPARATIVA")
    print("═" * 80)
    print(df.to_string(index=False))
    print("═" * 80)
    df.to_csv("results_summary.csv", index=False)
    print("  → Guardado: results_summary.csv")
    return df