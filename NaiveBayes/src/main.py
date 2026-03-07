import json
from modelo import entrenar, predecir, leave_one_out

# lee el archivo JSON como lista de registros directamente
with open('../data/iris.json', encoding='utf-8') as archivo:
    registros = json.load(archivo)

# detecta la estructura del dataset automaticamente
nombre_campo_clase  = list(registros[0].keys())[-1]
nombres_features    = [campo for campo in registros[0].keys() if campo != nombre_campo_clase]
clases_posibles     = list({registro[nombre_campo_clase] for registro in registros})
tipo_por_feature    = {
    feature: 'continuo' if isinstance(registros[0][feature], (int, float)) else 'discreto'
    for feature in nombres_features
}
valores_por_feature = {
    feature: list({registro[feature] for registro in registros})
    for feature in nombres_features
    if tipo_por_feature[feature] == 'discreto'
}

if __name__ == "__main__":

    prior, likelihood, gaussianos = entrenar(
        registros, nombre_campo_clase, nombres_features,
        clases_posibles, tipo_por_feature, valores_por_feature
    )

    aciertos_entrenamiento = sum(
        1 for registro in registros
        if predecir(registro, prior, likelihood, gaussianos, nombres_features, clases_posibles, tipo_por_feature)[0] == registro[nombre_campo_clase]
    )

    exactitud_entrenamiento = round(aciertos_entrenamiento / len(registros) * 100, 2)
    exactitud_loo           = round(leave_one_out(registros, nombre_campo_clase, nombres_features, clases_posibles, tipo_por_feature, valores_por_feature) * 100, 2)

    print("\n--- EVALUACION ---")
    print(f"Exactitud entrenamiento : {exactitud_entrenamiento} %")
    print(f"Exactitud Leave One Out : {exactitud_loo} %")

    while True:
        print("\nIngresa valores para predecir:")
        muestra_usuario = {}

        for feature in nombres_features:
            if tipo_por_feature[feature] == 'discreto':
                opciones = valores_por_feature[feature]
                print(f"  {feature}: {opciones}")
                while True:
                    valor_ingresado = input("  > ").strip().capitalize()
                    if valor_ingresado in opciones:
                        muestra_usuario[feature] = valor_ingresado
                        break
                    print("  Valor invalido, intenta de nuevo")
            else:
                print(f"  {feature} (numero):")
                while True:
                    try:
                        muestra_usuario[feature] = float(input("  > ").strip())
                        break
                    except ValueError:
                        print("  Debe ser un numero, intenta de nuevo")

        clase_predicha, probabilidades = predecir(
            muestra_usuario, prior, likelihood, gaussianos,
            nombres_features, clases_posibles, tipo_por_feature
        )

        if clase_predicha is None:
            print("No se pudo clasificar la muestra")
        else:
            print(f"\nResultado: {clase_predicha}")
            for clase in clases_posibles:
                print(f"  {clase} = {round(probabilidades[clase] * 100, 2)} %")

        if input("\nOtro s/n: ").strip().lower() != 's':
            break