# Naive Bayes - Prediccion de Juego

Clasificador Naive Bayes que predice si se jugara o no dependiendo de las condiciones climaticas.

## Archivos

- `naiveBayes.py` — clasificador principal
- `data/data_set.json` — dataset de entrenamiento (60 registros)
- `data/data_real.json` — dataset alternativo (54 registros)

## Como funciona

1. Lee el dataset desde un JSON
2. Detecta automaticamente los features, valores posibles y clases
3. Entrena el modelo calculando probabilidades con suavizado de Laplace
4. Evalua la exactitud con Leave-One-Out Cross Validation
5. Permite ingresar nuevos casos para predecir interactivamente

## Como correrlo

```bash
python naiveBayes.py
```

Para usar otro dataset, cambiar la ruta en la linea 4 del codigo:

```python
with open('data/data_real.json', encoding='utf-8') as archivo:
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

Lista de objetos JSON donde el **ultimo campo** es la clase a predecir:

```json
[
  {
    "clima": "Soleado",
    "temperatura": "Calor",
    "humedad": "Alta",
    "viento": "Debil",
    "juego": "No"
  },
  {
    "clima": "Nublado",
    "temperatura": "Frio",
    "humedad": "Baja",
    "viento": "Debil",
    "juego": "Si"
  }
]
```
