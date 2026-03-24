import os
from utils import cargar_dataset, normalizar, normalizar_muestra, pedir_k, pedir_muestra, mostrar_resultado
from knn_pesos import predecir, leave_one_out

# ─────────────────────────────────────────────────────────────
# DATASETS DISPONIBLES
# ruta construida relativa al archivo .py, no al directorio de ejecucion
# ─────────────────────────────────────────────────────────────

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

DATASETS = {
    '1': (os.path.join(BASE, 'iris.json'), 'Iris'),
    '2': (os.path.join(BASE, 'wdbc.json'), 'Breast Cancer Wisconsin'),
}

# ─────────────────────────────────────────────────────────────
# SELECCION DE DATASET
# ─────────────────────────────────────────────────────────────

print("=== KNN CON PESOS POR DISTANCIA ===")
print("  wi = (dk - di) / (dk - d1)")
print("\nSelecciona el dataset:")
for clave, (_, nombre) in DATASETS.items():
    print(f"  {clave}) {nombre}")

while True:
    opcion = input("  > ").strip()
    if opcion in DATASETS:
        ruta_dataset, nombre_dataset = DATASETS[opcion]
        break
    print("  Opcion invalida")

# el codigo detecta automaticamente clase, features y clases posibles
registros, nombre_campo_clase, nombres_features, clases_posibles = cargar_dataset(ruta_dataset)

print(f"\nDataset cargado : {nombre_dataset}")
print(f"Instancias      : {len(registros)}")
print(f"Features        : {len(nombres_features)}")
print(f"Clases          : {clases_posibles}")

# normalizar para trabajar en espacio n-dimensional normalizado
registros_norm, minimos, maximos = normalizar(registros, nombres_features)

# ─────────────────────────────────────────────────────────────
# SELECCION DE K
# ─────────────────────────────────────────────────────────────

k = pedir_k(len(registros) - 1)

# ─────────────────────────────────────────────────────────────
# EVALUACION
# ─────────────────────────────────────────────────────────────

aciertos = sum(
    1 for r in registros_norm
    if predecir(r, registros_norm, k, nombre_campo_clase, nombres_features, clases_posibles)[0]
       == r[nombre_campo_clase]
)
exactitud_entrenamiento = round(aciertos / len(registros_norm) * 100, 2)

print("\nCalculando Leave-One-Out...")
exactitud_loo = round(
    leave_one_out(registros_norm, k, nombre_campo_clase, nombres_features, clases_posibles) * 100, 2
)

print("\n--- EVALUACION ---")
print(f"Dataset                 : {nombre_dataset}")
print(f"K                       : {k}")
print(f"Exactitud entrenamiento : {exactitud_entrenamiento} %")
print(f"Exactitud Leave-One-Out : {exactitud_loo} %")

# ─────────────────────────────────────────────────────────────
# PREDICCION INTERACTIVA
# ─────────────────────────────────────────────────────────────

while True:
    muestra = pedir_muestra(nombres_features)
    muestra_norm = normalizar_muestra(muestra, minimos, maximos, nombres_features)

    clase_predicha, votos = predecir(
        muestra_norm, registros_norm, k,
        nombre_campo_clase, nombres_features, clases_posibles
    )
    mostrar_resultado(clase_predicha, votos, clases_posibles)

    if input("\nOtro? s/n: ").strip().lower() != 's':
        break
