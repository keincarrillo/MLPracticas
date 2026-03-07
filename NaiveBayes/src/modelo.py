from collections import defaultdict
from utils import calcular_media, calcular_desviacion, gaussiana

# entrenar(registros_entrenamiento: list[dict], nombre_campo_clase: str, nombres_features: list, clases_posibles: list, tipo_por_feature: dict, valores_por_feature: dict) -> (probabilidad_prior: dict, probabilidad_likelihood: dict, parametros_gaussianos: dict)
# calcula prior y likelihood para discretos, y media/desviacion para continuos con suavizado de Laplace
def entrenar(registros_entrenamiento, nombre_campo_clase, nombres_features, clases_posibles, tipo_por_feature, valores_por_feature):

    total_registros                = len(registros_entrenamiento)
    conteo_por_clase               = defaultdict(int)
    valores_continuos_por_clase    = defaultdict(list)
    conteo_por_clase_feature_valor = defaultdict(int)

    for registro in registros_entrenamiento:
        clase_actual = registro[nombre_campo_clase]
        conteo_por_clase[clase_actual] += 1
        for feature in nombres_features:
            if tipo_por_feature[feature] == 'continuo':
                valores_continuos_por_clase[(clase_actual, feature)].append(registro[feature])
            else:
                conteo_por_clase_feature_valor[(clase_actual, feature, registro[feature])] += 1

    # P(clase) con suavizado de Laplace
    probabilidad_prior = {
        clase: (conteo_por_clase[clase] + 1) / (total_registros + len(clases_posibles))
        for clase in clases_posibles
    }

    # P(valor | clase) para features discretos con suavizado de Laplace
    probabilidad_likelihood = {}
    for clase in clases_posibles:
        for feature in nombres_features:
            if tipo_por_feature[feature] == 'discreto':
                for valor in valores_por_feature[feature]:
                    clave = (clase, feature, valor)
                    probabilidad_likelihood[clave] = (
                        conteo_por_clase_feature_valor[clave] + 1
                    ) / (conteo_por_clase[clase] + len(valores_por_feature[feature]))

    # media y desviacion por (clase, feature) para features continuos
    parametros_gaussianos = {}
    for clase in clases_posibles:
        for feature in nombres_features:
            if tipo_por_feature[feature] == 'continuo':
                valores    = valores_continuos_por_clase[(clase, feature)]
                media      = calcular_media(valores)
                desviacion = calcular_desviacion(valores, media)
                parametros_gaussianos[(clase, feature)] = (media, desviacion)

    return probabilidad_prior, probabilidad_likelihood, parametros_gaussianos


# predecir(muestra: dict, probabilidad_prior: dict, probabilidad_likelihood: dict, parametros_gaussianos: dict, nombres_features: list, clases_posibles: list, tipo_por_feature: dict) -> (clase_predicha: str | None, probabilidades_normalizadas: dict)
# retorna la clase con mayor probabilidad y el porcentaje de cada clase para una muestra dada
def predecir(muestra, probabilidad_prior, probabilidad_likelihood, parametros_gaussianos, nombres_features, clases_posibles, tipo_por_feature):

    puntaje_por_clase = {}

    for clase in clases_posibles:
        probabilidad_acumulada = probabilidad_prior[clase]
        for feature in nombres_features:
            if tipo_por_feature[feature] == 'discreto':
                clave = (clase, feature, muestra[feature])
                probabilidad_acumulada *= probabilidad_likelihood.get(clave, 0)
            else:
                media, desviacion = parametros_gaussianos[(clase, feature)]
                probabilidad_acumulada *= gaussiana(muestra[feature], media, desviacion)
        puntaje_por_clase[clase] = probabilidad_acumulada

    suma_puntajes = sum(puntaje_por_clase.values())

    if suma_puntajes == 0:
        return None, {clase: 0 for clase in clases_posibles}

    probabilidades_normalizadas = {
        clase: puntaje_por_clase[clase] / suma_puntajes
        for clase in clases_posibles
    }

    clase_predicha = max(probabilidades_normalizadas, key=probabilidades_normalizadas.get)

    return clase_predicha, probabilidades_normalizadas


# leave_one_out(registros: list[dict], nombre_campo_clase: str, nombres_features: list, clases_posibles: list, tipo_por_feature: dict, valores_por_feature: dict) -> exactitud: float
# entrena y evalua N veces dejando un registro fuera como prueba, retorna proporcion de aciertos
def leave_one_out(registros, nombre_campo_clase, nombres_features, clases_posibles, tipo_por_feature, valores_por_feature):

    predicciones_correctas = 0

    for indice in range(len(registros)):
        registros_train = registros[:indice] + registros[indice + 1:]
        registro_test   = registros[indice]

        prior_loo, likelihood_loo, gaussianos_loo = entrenar(
            registros_train, nombre_campo_clase, nombres_features,
            clases_posibles, tipo_por_feature, valores_por_feature
        )
        clase_predicha, _ = predecir(
            registro_test, prior_loo, likelihood_loo, gaussianos_loo,
            nombres_features, clases_posibles, tipo_por_feature
        )

        if clase_predicha == registro_test[nombre_campo_clase]:
            predicciones_correctas += 1

    return predicciones_correctas / len(registros)