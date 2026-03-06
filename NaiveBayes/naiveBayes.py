import json
from collections import defaultdict

# lee el archivo JSON y guarda todos los registros en una lista de diccionarios
with open('data/data_set.json', encoding='utf-8') as archivo:
    dataset = json.load(archivo)

# soporta dos formatos: {config, datos} o directamente una lista de registros
registros = dataset['datos'] if isinstance(dataset, dict) and 'datos' in dataset else dataset

# el ultimo campo del primer registro es la variable a predecir
nombre_campo_clase  = list(registros[0].keys())[-1]

# todos los campos excepto la clase son los atributos de entrada (features)
nombres_features    = [campo for campo in registros[0].keys() if campo != nombre_campo_clase]

# valores posibles de cada feature, detectados recorriendo todo el dataset
valores_por_feature = {
    feature: list({registro[feature] for registro in registros})
    for feature in nombres_features
}

# clases posibles (ej. Si / No), detectadas recorriendo todo el dataset
clases_posibles     = list({registro[nombre_campo_clase] for registro in registros})


# recibe el dataset y retorna prior y likelihood calculados con suavizado de Laplace
# entrenar(registros_entrenamiento: list[dict]) -> (probabilidad_prior: dict, probabilidad_likelihood: dict)
def entrenar(registros_entrenamiento):

    total_registros = len(registros_entrenamiento)

    # cuantas veces aparece cada clase en el dataset
    conteo_por_clase = defaultdict(int)

    # cuantas veces aparece cada combinacion (clase, feature, valor)
    conteo_por_clase_feature_valor = defaultdict(int)

    # recorre cada registro y acumula los conteos
    for registro in registros_entrenamiento:
        clase_actual = registro[nombre_campo_clase]
        conteo_por_clase[clase_actual] += 1
        for feature in nombres_features:
            clave = (clase_actual, feature, registro[feature])
            conteo_por_clase_feature_valor[clave] += 1

    # P(clase): probabilidad de cada clase en el dataset
    # +1 y +num_clases es el suavizado de Laplace para evitar probabilidades de cero
    probabilidad_prior = {
        clase: (conteo_por_clase[clase] + 1) / (total_registros + len(clases_posibles))
        for clase in clases_posibles
    }

    # P(valor | clase): probabilidad de cada valor dado que el registro pertenece a cierta clase
    # +1 y +num_valores es el suavizado de Laplace
    probabilidad_likelihood = {}
    for clase in clases_posibles:
        for feature in nombres_features:
            for valor in valores_por_feature[feature]:
                clave = (clase, feature, valor)
                probabilidad_likelihood[clave] = (
                    conteo_por_clase_feature_valor[clave] + 1
                ) / (conteo_por_clase[clase] + len(valores_por_feature[feature]))

    return probabilidad_prior, probabilidad_likelihood


# recibe una muestra sin clase, retorna la clase predicha y el porcentaje de cada clase
# predecir(muestra: dict, probabilidad_prior: dict, probabilidad_likelihood: dict) -> (clase_predicha: str | None, probabilidades_normalizadas: dict)
def predecir(muestra, probabilidad_prior, probabilidad_likelihood):

    puntaje_por_clase = {}

    for clase in clases_posibles:
        # empieza con P(clase) y multiplica P(valor | clase) por cada feature
        probabilidad_acumulada = probabilidad_prior[clase]
        for feature in nombres_features:
            clave = (clase, feature, muestra[feature])
            probabilidad_acumulada *= probabilidad_likelihood.get(clave, 0)
        puntaje_por_clase[clase] = probabilidad_acumulada

    # normaliza los puntajes para que las probabilidades sumen 1
    suma_puntajes = sum(puntaje_por_clase.values())

    # si todos los puntajes son cero no se puede clasificar
    if suma_puntajes == 0:
        return None, {clase: 0 for clase in clases_posibles}

    probabilidades_normalizadas = {
        clase: puntaje_por_clase[clase] / suma_puntajes
        for clase in clases_posibles
    }

    # retorna la clase con mayor probabilidad
    clase_predicha = max(probabilidades_normalizadas, key=probabilidades_normalizadas.get)

    return clase_predicha, probabilidades_normalizadas


# entrena y evalua N veces dejando un registro fuera como prueba en cada iteracion
# retorna la proporcion de predicciones correctas como estimacion sobre datos nuevos
# leave_one_out(registros: list[dict]) -> exactitud: float
def leave_one_out(registros):

    predicciones_correctas = 0

    for indice in range(len(registros)):
        # usa todos los registros menos el actual para entrenar
        registros_train = registros[:indice] + registros[indice + 1:]
        registro_test   = registros[indice]

        # entrena sin el registro de prueba y predice sobre el
        prior_loo, likelihood_loo = entrenar(registros_train)
        clase_predicha, _         = predecir(registro_test, prior_loo, likelihood_loo)

        if clase_predicha == registro_test[nombre_campo_clase]:
            predicciones_correctas += 1

    return predicciones_correctas / len(registros)


if __name__ == "__main__":

    # entrena el modelo con todos los registros
    prior, likelihood = entrenar(registros)

    # cuenta cuantos registros del dataset se clasifican correctamente
    aciertos_entrenamiento = sum(
        1 for registro in registros
        if predecir(registro, prior, likelihood)[0] == registro[nombre_campo_clase]
    )

    # convierte los aciertos a porcentaje
    exactitud_entrenamiento = round(aciertos_entrenamiento / len(registros) * 100, 2)
    exactitud_loo           = round(leave_one_out(registros) * 100, 2)

    print("\n--- EVALUACION ---")
    print(f"Exactitud entrenamiento : {exactitud_entrenamiento} %")
    print(f"Exactitud Leave One Out : {exactitud_loo} %")

    # bucle para predecir nuevos casos hasta que el usuario decida salir
    while True:
        print("\nIngresa valores para predecir:")
        muestra_usuario = {}

        # pide al usuario el valor de cada feature y valida que sea una opcion valida
        for feature in nombres_features:
            opciones = valores_por_feature[feature]
            print(f"  {feature}: {opciones}")

            while True:
                valor_ingresado = input("  > ").strip().capitalize()
                if valor_ingresado in opciones:
                    muestra_usuario[feature] = valor_ingresado
                    break
                print("  Valor invalido, intenta de nuevo")

        # clasifica la muestra ingresada
        clase_predicha, probabilidades = predecir(muestra_usuario, prior, likelihood)

        if clase_predicha is None:
            print("No se pudo clasificar la muestra")
        else:
            print(f"\nResultado: {clase_predicha}")
            # muestra el porcentaje de cada clase
            for clase in clases_posibles:
                print(f"  {clase} = {round(probabilidades[clase] * 100, 2)} %")

        if input("\nOtro s/n: ").strip().lower() != 's':
            break