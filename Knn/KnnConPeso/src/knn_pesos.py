from utils import calcular_distancia

# ─────────────────────────────────────────────────────────────
# KNN CON PESOS POR DISTANCIA
# formula de clase: wi = (dk - di) / (dk - d1)
#   d1 = distancia al vecino mas cercano
#   dk = distancia al vecino mas lejano (entre los k)
#   di = distancia al vecino i
# el vecino mas cercano recibe peso 1, el mas lejano peso 0
# ─────────────────────────────────────────────────────────────

# calcular_pesos(vecinos: list[tuple]) -> dict {id(registro): peso}
# aplica la formula wi = (dk - di) / (dk - d1)
# si todos estan a la misma distancia reparte peso uniforme
def calcular_pesos(vecinos):
    d1 = vecinos[0][1]   # distancia minima
    dk = vecinos[-1][1]  # distancia maxima

    pesos = {}
    if dk == d1:
        # todos equidistantes -> peso uniforme
        peso_uniforme = 1 / len(vecinos)
        for registro, _ in vecinos:
            pesos[id(registro)] = peso_uniforme
    else:
        for registro, di in vecinos:
            pesos[id(registro)] = (dk - di) / (dk - d1)

    return pesos


# predecir(muestra, registros_entrenamiento, k, nombre_campo_clase, nombres_features, clases_posibles)
#   -> (clase_predicha: str, votos_ponderados: dict)
# busca los k vecinos mas cercanos, calcula sus pesos y suma por clase
def predecir(muestra, registros_entrenamiento, k, nombre_campo_clase, nombres_features, clases_posibles):

    # distancia de la muestra a cada registro
    distancias = [
        (registro, calcular_distancia(muestra, registro, nombres_features))
        for registro in registros_entrenamiento
    ]

    # k vecinos mas cercanos ordenados por distancia
    vecinos = sorted(distancias, key=lambda par: par[1])[:k]

    # peso de cada vecino segun su distancia
    pesos = calcular_pesos(vecinos)

    # sumar pesos por clase
    votos_ponderados = {clase: 0.0 for clase in clases_posibles}
    for registro, _ in vecinos:
        votos_ponderados[registro[nombre_campo_clase]] += pesos[id(registro)]

    clase_predicha = max(votos_ponderados, key=votos_ponderados.get)
    return clase_predicha, votos_ponderados


# leave_one_out(registros, k, nombre_campo_clase, nombres_features, clases_posibles) -> float
# deja un registro fuera como prueba N veces, retorna proporcion de aciertos
def leave_one_out(registros, k, nombre_campo_clase, nombres_features, clases_posibles):
    aciertos = 0
    for i in range(len(registros)):
        train = registros[:i] + registros[i + 1:]
        test  = registros[i]
        clase_predicha, _ = predecir(test, train, k, nombre_campo_clase, nombres_features, clases_posibles)
        if clase_predicha == test[nombre_campo_clase]:
            aciertos += 1
    return aciertos / len(registros)
