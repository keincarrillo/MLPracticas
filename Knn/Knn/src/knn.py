from utils import calcular_distancia

# ─────────────────────────────────────────────────────────────
# KNN CLASICO  (voto por mayoria simple)
# ─────────────────────────────────────────────────────────────

# predecir(muestra, registros_entrenamiento, k, nombre_campo_clase, nombres_features, clases_posibles)
#   -> (clase_predicha: str, votos: dict)
# busca los k vecinos mas cercanos y la clase con mas votos gana
def predecir(muestra, registros_entrenamiento, k, nombre_campo_clase, nombres_features, clases_posibles):

    # distancia de la muestra a cada registro
    distancias = [
        (registro, calcular_distancia(muestra, registro, nombres_features))
        for registro in registros_entrenamiento
    ]

    # k vecinos mas cercanos
    vecinos = sorted(distancias, key=lambda par: par[1])[:k]

    # voto por mayoria simple: cada vecino vale 1
    votos = {clase: 0 for clase in clases_posibles}
    for registro, _ in vecinos:
        votos[registro[nombre_campo_clase]] += 1

    clase_predicha = max(votos, key=votos.get)
    return clase_predicha, votos


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
