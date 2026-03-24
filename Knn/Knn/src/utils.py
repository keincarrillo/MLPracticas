import json
from math import sqrt

# ─────────────────────────────────────────────────────────────
# CARGA Y DETECCION AUTOMATICA DEL DATASET
# ─────────────────────────────────────────────────────────────

# cargar_dataset(ruta: str) -> (registros, nombre_campo_clase, nombres_features, clases_posibles)
# lee cualquier JSON con lista de registros y detecta automaticamente:
#   - el campo clase (ultimo campo de cada registro)
#   - los features (todos los campos numericos excepto la clase)
#   - las clases posibles (valores unicos del campo clase)
def cargar_dataset(ruta):
    with open(ruta, encoding='utf-8') as archivo:
        registros = json.load(archivo)

    nombre_campo_clase = list(registros[0].keys())[-1]
    nombres_features   = [
        campo for campo in registros[0].keys()
        if campo != nombre_campo_clase and isinstance(registros[0][campo], (int, float))
    ]
    clases_posibles = list({registro[nombre_campo_clase] for registro in registros})

    return registros, nombre_campo_clase, nombres_features, clases_posibles


# ─────────────────────────────────────────────────────────────
# NORMALIZACION MIN-MAX
# ─────────────────────────────────────────────────────────────

# normalizar(registros, nombres_features) -> (registros_normalizados, minimos, maximos)
# escala cada feature entre 0 y 1 para trabajar en espacio normalizado
def normalizar(registros, nombres_features):
    minimos = {f: min(r[f] for r in registros) for f in nombres_features}
    maximos = {f: max(r[f] for r in registros) for f in nombres_features}

    registros_normalizados = []
    for registro in registros:
        copia = dict(registro)
        for f in nombres_features:
            rango    = maximos[f] - minimos[f]
            copia[f] = (registro[f] - minimos[f]) / rango if rango > 0 else 0.0
        registros_normalizados.append(copia)

    return registros_normalizados, minimos, maximos


# normalizar_muestra(muestra, minimos, maximos, nombres_features) -> dict
# aplica la misma normalizacion del entrenamiento a una muestra nueva
def normalizar_muestra(muestra, minimos, maximos, nombres_features):
    copia = dict(muestra)
    for f in nombres_features:
        rango    = maximos[f] - minimos[f]
        copia[f] = (muestra[f] - minimos[f]) / rango if rango > 0 else 0.0
    return copia


# ─────────────────────────────────────────────────────────────
# DISTANCIA EUCLIDIANA
# ─────────────────────────────────────────────────────────────

# calcular_distancia(a, b, nombres_features) -> float
def calcular_distancia(a, b, nombres_features):
    return sqrt(sum((a[f] - b[f]) ** 2 for f in nombres_features))


# ─────────────────────────────────────────────────────────────
# ENTRADA Y SALIDA
# ─────────────────────────────────────────────────────────────

# pedir_k(max_k: int) -> int
def pedir_k(max_k):
    print(f"\nIngresa el valor de K (1 - {max_k}):")
    while True:
        try:
            k = int(input("  > ").strip())
            if 1 <= k <= max_k:
                return k
            print(f"  K debe estar entre 1 y {max_k}")
        except ValueError:
            print("  Debe ser un numero entero")


# pedir_muestra(nombres_features: list) -> dict
def pedir_muestra(nombres_features):
    print("\nIngresa valores para predecir:")
    muestra = {}
    for feature in nombres_features:
        print(f"  {feature} (numero):")
        while True:
            try:
                muestra[feature] = float(input("  > ").strip())
                break
            except ValueError:
                print("  Debe ser un numero, intenta de nuevo")
    return muestra


# mostrar_resultado(clase_predicha, votos, clases_posibles) -> None
def mostrar_resultado(clase_predicha, votos, clases_posibles):
    suma = sum(votos.values())
    print(f"\nResultado: {clase_predicha}")
    for clase in clases_posibles:
        porcentaje = round(votos[clase] / suma * 100, 2) if suma > 0 else 0.0
        print(f"  {clase} = {porcentaje} %")
