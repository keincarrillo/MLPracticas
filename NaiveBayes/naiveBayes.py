import json
from collections import defaultdict

# cargar datos desde archivo json
with open('data_real.json', encoding='utf-8') as f:
    datos = json.load(f)

# clases posibles
CLASES = ['Si', 'No']

# caracteristicas que usa el modelo
FEATURES = ['clima', 'temperatura', 'humedad', 'viento']


# entrenar modelo naive bayes sin suavizado
def entrenar(datos):
    total = len(datos)

    # contar clases
    conteo_clase = defaultdict(int)

    # contar valores por clase
    conteo = defaultdict(int)

    for d in datos:
        c = d['juego']
        conteo_clase[c] += 1
        for f in FEATURES:
            conteo[(c, f, d[f])] += 1

    # calcular probabilidad de cada clase
    prior = {c: conteo_clase[c] / total for c in CLASES}

    # calcular probabilidad de cada valor dado la clase
    likelihood = {}
    for c in CLASES:
        for f in FEATURES:
            for d in datos:
                val = d[f]
                if conteo_clase[c] == 0:
                    likelihood[(c, f, val)] = 0
                else:
                    likelihood[(c, f, val)] = (
                        conteo[(c, f, val)] / conteo_clase[c]
                    )

    return prior, likelihood


# predecir clase de una muestra
def predecir(muestra, prior, likelihood):
    scores = {}

    # multiplicar probabilidades
    for c in CLASES:
        p = prior[c]
        for f in FEATURES:
            p *= likelihood.get((c, f, muestra[f]), 0)
        scores[c] = p

    total = sum(scores.values())

    # si todo es cero no se puede clasificar
    if total == 0:
        return None, {c: 0 for c in CLASES}

    # normalizar probabilidades
    probs = {c: scores[c] / total for c in CLASES}

    # elegir clase con mayor probabilidad
    pred = max(probs, key=probs.get)

    return pred, probs


# validacion leave one out
def leave_one_out(datos):
    correctos = 0

    for i in range(len(datos)):
        train = datos[:i] + datos[i+1:]
        test = datos[i]

        prior, likelihood = entrenar(train)
        pred, _ = predecir(test, prior, likelihood)

        if pred == test['juego']:
            correctos += 1

    return correctos / len(datos)


# programa principal
if __name__ == "__main__":

    # entrenar con todos los datos
    prior, likelihood = entrenar(datos)

    # calcular exactitud en entrenamiento
    correctos = sum(
        1 for d in datos
        if predecir(d, prior, likelihood)[0] == d['juego']
    )

    print("\n=== EVALUACION ===")
    print("Exactitud entrenamiento:", correctos / len(datos))
    print("Exactitud Leave One Out:", leave_one_out(datos))

    # modo interactivo para predecir nuevos casos
    while True:
        print("\nIngresa valores:")
        caso = {}
        for f in FEATURES:
            caso[f] = input(f"{f}: ").capitalize()

        pred, probs = predecir(caso, prior, likelihood)

        if pred is None:
            print("No se pudo clasificar")
        else:
            print("\nResultado:", pred)
            print("Si =", round(probs['Si'] * 100, 2), "%")
            print("No =", round(probs['No'] * 100, 2), "%")

        if input("\nOtro s n: ").lower() != 's':
            break