# visualizaciones.py
# Módulo de gráficas para comparar y analizar los 6 modelos MLP sobre Iris.
# Mantiene el mismo estilo visual oscuro de la práctica anterior.
#
# Funciones disponibles:
#   plot_learning_curves      → pérdida, exactitud y norma de pesos por época
#   plot_confusion_matrices   → matrices de confusión multiclase (3x3)
#   plot_metrics_comparison   → barras comparativas de métricas en test
#   plot_training_times       → tiempos de entrenamiento en milisegundos
#   plot_pso_swarm            → fitness global, diversidad y distribución pbest
#   plot_weight_distributions → histograma de pesos finales de cada modelo
#   build_summary_table       → tabla resumen + guardado en CSV

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Nombres de las clases: se asignan dinámicamente desde main.py
# via set_class_names() para que las gráficas reflejen el dataset real.
# Valor por defecto genérico en caso de que no se llame a set_class_names.
_CLASS_NAMES = ["Clase 0", "Clase 1", "Clase 2"]

# set_class_names(names: list[str]) -> None
# Permite que main.py inyecte los nombres reales de las clases detectadas
# en el dataset JSON. Se llama UNA VEZ antes de generar cualquier gráfica.
# Así las matrices de confusión, reportes, etc. usan "setosa" en lugar de "Clase 0".
def set_class_names(names):
    global _CLASS_NAMES
    _CLASS_NAMES = list(names)

# Paleta de colores: un color por modelo
COLORS = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261", "#264653"]

# _setup_style() -> None
# Aplica el tema oscuro de matplotlib.
# Se llama al inicio de cada función para mantener consistencia visual.
def _setup_style():
    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor":   "#161b22",
        "axes.edgecolor":   "#30363d",
        "axes.labelcolor":  "#c9d1d9",
        "xtick.color":      "#c9d1d9",
        "ytick.color":      "#c9d1d9",
        "text.color":       "#c9d1d9",
        "grid.color":       "#21262d",
        "grid.linewidth":   0.5,
        "legend.facecolor": "#161b22",
        "legend.edgecolor": "#30363d",
        "font.family":      "monospace",
    })

# plot_learning_curves(results_list: list[dict]) -> None
# Genera 3 subgráficas por fila:
#   izquierda : entropía cruzada categórica por época
#   centro    : exactitud (accuracy) por época
#   derecha   : norma total de los pesos (mide si los pesos crecen o se estabilizan)
# Permite detectar sobreajuste (train sube, test baja) y convergencia.
def plot_learning_curves(results_list):
    _setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Curvas de Aprendizaje — Pérdida, Exactitud y Norma de Pesos",
                 fontsize=13, color="#c9d1d9", y=1.02)

    for i, res in enumerate(results_list):
        col = COLORS[i % len(COLORS)]
        ep  = range(1, len(res["history"]) + 1)
        axes[0].plot(ep, res["history"],          color=col, lw=1.5, label=res["name"])
        axes[1].plot(ep, res["accuracy_history"],  color=col, lw=1.5)
        axes[2].plot(ep, res["weight_history"],    color=col, lw=1.5)

    labels = ["Pérdida (Cross-Entropy)", "Exactitud (train)", "Norma total de pesos"]
    for ax, title in zip(axes, labels):
        ax.set_xlabel("Época / Generación")
        ax.set_title(title, color="#c9d1d9")
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print("  → Guardado: learning_curves.png")

# plot_confusion_matrices(results_list: list[dict]) -> None
# Matrices de confusión 3×3 (una por clase de Iris: Setosa, Versicolor, Virginica).
# Permite ver qué clases se confunden entre sí; Iris tiene clases difíciles de separar
# (Versicolor vs Virginica son las más parecidas).
def plot_confusion_matrices(results_list):
    _setup_style()
    n    = len(results_list)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5))
    axes = np.array(axes).flatten()
    fig.suptitle("Matrices de Confusión (Conjunto de Prueba)", fontsize=13, color="#c9d1d9")

    for i, res in enumerate(results_list):
        cm  = confusion_matrix(res["y_test"], res["y_pred_test"])
        col = COLORS[i % len(COLORS)]
        cmap = sns.light_palette(col, as_cmap=True)
        sns.heatmap(cm, ax=axes[i], annot=True, fmt="d", cmap=cmap,
                    linewidths=0.5, linecolor="#21262d",
                    annot_kws={"size": 14, "color": "#0d1117"},
                    cbar=False)
        axes[i].set_title(res["name"], fontsize=9, color="#c9d1d9")
        axes[i].set_xlabel("Predicho", fontsize=8)
        axes[i].set_ylabel("Real", fontsize=8)
        axes[i].set_xticklabels(_CLASS_NAMES, fontsize=8, rotation=15)
        axes[i].set_yticklabels(_CLASS_NAMES, fontsize=8, rotation=0)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print("  → Guardado: confusion_matrices.png")

# plot_metrics_comparison(results_list: list[dict]) -> None
# Barras agrupadas comparando accuracy, precision, recall, f1 y mcc en test.
# Nota: no se incluye AUC-ROC porque para multiclase requiere el enfoque OvR
# y queremos mantener la comparativa simple y consistente.
def plot_metrics_comparison(results_list):
    _setup_style()
    metrics = ["accuracy", "precision", "recall", "f1", "mcc"]
    labels  = ["Accuracy", "Precision", "Recall", "F1", "MCC"]
    x     = np.arange(len(metrics))
    width = 0.8 / len(results_list)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, res in enumerate(results_list):
        vals   = [res["test"].get(m, 0) for m in metrics]
        offset = (i - len(results_list) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width=width * 0.9,
                        color=COLORS[i % len(COLORS)], label=res["name"],
                        alpha=0.85, edgecolor="#0d1117", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=5.5, color="#c9d1d9")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title("Comparación de Métricas en Test (Iris)", fontsize=13, color="#c9d1d9")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("metrics_comparison.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print("  → Guardado: metrics_comparison.png")

# plot_training_times(results_list: list[dict]) -> None
# Barras horizontales con el tiempo de entrenamiento en milisegundos.
# PSO suele ser más lento que gradiente porque evalúa N partículas por generación.
def plot_training_times(results_list):
    _setup_style()
    names  = [r["name"] for r in results_list]
    times  = [r["train_time"] * 1000 for r in results_list]
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
    plt.close()
    print("  → Guardado: training_times.png")

# plot_pso_swarm(results_list: list[dict]) -> None
# Análisis del enjambre para modelos PSO: fitness global, diversidad y pbest.
# Si no hay modelos PSO en la lista, la función no hace nada.
def plot_pso_swarm(results_list):
    pso_results = [r for r in results_list if "pso" in r]
    if not pso_results:
        return

    _setup_style()
    n    = len(pso_results)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4.5 * n))
    if n == 1:
        axes = [axes]
    fig.suptitle("Análisis del Enjambre PSO — MLP Iris", fontsize=13, color="#c9d1d9")

    for row, res in enumerate(pso_results):
        p   = res["pso"]
        col = COLORS[results_list.index(res) % len(COLORS)]
        epochs = range(1, len(p["swarm_fitness_history"]) + 1)

        # Fitness del mejor global: debe decrecer monotónicamente (siempre guarda el mejor)
        axes[row][0].plot(epochs, p["swarm_fitness_history"], color=col, lw=1.5)
        axes[row][0].set_title(f"{res['name']} — Fitness gbest", fontsize=9)
        axes[row][0].set_xlabel("Generación")
        axes[row][0].grid(True, alpha=0.3)

        # Diversidad: si cae a 0 rápidamente el enjambre convergió prematuramente
        axes[row][1].plot(epochs, res["model"].diversity_history, color="#f4a261", lw=1.5)
        axes[row][1].set_title(f"{res['name']} — Diversidad del enjambre", fontsize=9)
        axes[row][1].set_xlabel("Generación")
        axes[row][1].grid(True, alpha=0.3)

        # Distribución del fitness personal al final del entrenamiento
        axes[row][2].hist(p["personal_best_fitness"], bins=12,
                          color=col, edgecolor="#0d1117", alpha=0.85)
        axes[row][2].axvline(p["best_fitness"], color="#fff", linestyle="--", lw=1.2,
                             label=f"Mejor={p['best_fitness']:.4f}")
        axes[row][2].axvline(p["mean_pbest"], color="#e9c46a", linestyle="--", lw=1.2,
                             label=f"Media={p['mean_pbest']:.4f}")
        axes[row][2].set_title(f"{res['name']} — Distribución pbest", fontsize=9)
        axes[row][2].legend(fontsize=7)
        axes[row][2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pso_swarm_analysis.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print("  → Guardado: pso_swarm_analysis.png")

# plot_weight_distributions(results_list: list[dict]) -> None
# Histograma de todos los pesos de la red (todas las capas concatenadas).
# Distribución estrecha → pesos pequeños (menos riesgo de sobreajuste).
# Distribución ancha → pesos grandes (el modelo usa las características con fuerza).
def plot_weight_distributions(results_list):
    _setup_style()
    n    = len(results_list)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5))
    axes = np.array(axes).flatten()
    fig.suptitle("Distribución de Pesos Finales (todas las capas)", fontsize=13, color="#c9d1d9")

    for i, res in enumerate(results_list):
        col = COLORS[i % len(COLORS)]
        # Concatenar todos los pesos de todas las capas en un solo vector
        all_w = np.concatenate([W.flatten() for W in res["model"].weights])
        axes[i].hist(all_w, bins=25, color=col, edgecolor="#0d1117", alpha=0.85, density=True)
        axes[i].axvline(np.mean(all_w), color="#fff", lw=1.2, linestyle="--",
                        label=f"μ={np.mean(all_w):.3f}")
        axes[i].set_title(res["name"], fontsize=9, color="#c9d1d9")
        axes[i].legend(fontsize=7)
        axes[i].grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig("weight_distributions.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print("  → Guardado: weight_distributions.png")

# build_summary_table(results_list: list[dict]) -> pd.DataFrame
# Construye y muestra la tabla comparativa de todos los modelos.
# También guarda los resultados en CSV para análisis externo (Excel, etc.).
def build_summary_table(results_list):
    rows = []
    for res in results_list:
        row = {
            "Modelo":       res["name"],
            "Acc Train":    f"{res['train']['accuracy']:.4f}",
            "Acc Test":     f"{res['test']['accuracy']:.4f}",
            "F1 Test":      f"{res['test']['f1']:.4f}",
            "MCC Test":     f"{res['test']['mcc']:.4f}",
            "Tiempo(ms)":   f"{res['train_time'] * 1000:.1f}",
            "Convergencia": str(res["converged_epoch"]) if res["converged_epoch"] else "—",
        }
        if "pso" in res:
            row["PSO Partículas"] = str(res["pso"]["n_particles"])
            row["PSO Mejor Fit"]  = f"{res['pso']['best_fitness']:.5f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n" + "═" * 80)
    print("  TABLA RESUMEN COMPARATIVA — MLP IRIS")
    print("═" * 80)
    print(df.to_string(index=False))
    print("═" * 80)
    df.to_csv("results_summary.csv", index=False)
    print("  → Guardado: results_summary.csv")
    return df
