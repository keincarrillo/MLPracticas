# Naive Bayes - Prediccion de Juego

Clasificador Naive Bayes que predice si se jugara o no dependiendo de las condiciones climaticas.
Soporta features discretos (strings) y continuos (numeros) de forma automatica.

## Archivos

- `main.py` — carga del dataset, deteccion automatica y loop interactivo
- `modelo.py` — logica del clasificador (entrenar, predecir, leave_one_out)
- `utils.py` — funciones matematicas (media, desviacion, gaussiana)
- `data/data_set.json` — dataset de entrenamiento (60 registros, discreto)
- `data/data_real.json` — dataset alternativo (54 registros, discreto)
- `data/iris.json` — dataset iris (150 registros, continuo)

## Como funciona

1. Lee el dataset desde un JSON
2. Detecta automaticamente features, valores posibles y clases
3. Clasifica cada feature como discreto (str) o continuo (int/float)
4. Entrena el modelo con Laplace para discretos y Gaussiana para continuos
5. Evalua la exactitud con Leave-One-Out Cross Validation
6. Permite ingresar nuevos casos para predecir interactivamente

## Como correrlo

```bash
python3 src/main.py
```

Para usar otro dataset, cambiar la ruta en `main.py`:

```python
with open('data/iris.json', encoding='utf-8') as archivo:
```

## Ejemplo de salida

```
--- EVALUACION ---
Exactitud entrenamiento : 96.67 %
Exactitud Leave One Out : 90.0 %

Ingresa valores para predecir:
  clima: ['Soleado', 'Nublado', 'Lluvia']
  > Nublado
  temperatura: ['Calor', 'Templado', 'Frio']
  > Templado
  humedad: ['Alta', 'Normal', 'Baja']
  > Normal
  viento: ['Debil', 'Fuerte']
  > Debil

Resultado: Si
  Si = 89.34 %
  No = 10.66 %
```

## Formato del dataset

Lista de objetos JSON donde el **ultimo campo** es la clase a predecir.

Discreto:

```json
[
  {
    "clima": "Soleado",
    "temperatura": "Calor",
    "humedad": "Alta",
    "viento": "Debil",
    "juego": "No"
  }
]
```

Continuo:

```json
[
  {
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2,
    "especie": "setosa"
  }
]
```
