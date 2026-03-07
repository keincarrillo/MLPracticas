from math import sqrt, pi, exp

# calcular_media(valores: list[float]) -> float
# retorna el promedio de una lista de numeros
def calcular_media(valores):
    return sum(valores) / len(valores)

# calcular_desviacion(valores: list[float], media: float) -> float
# retorna la desviacion estandar de una lista de numeros
def calcular_desviacion(valores, media):
    varianza = sum((x - media) ** 2 for x in valores) / len(valores)
    return sqrt(varianza) if varianza > 0 else 1e-9

# gaussiana(x: float, media: float, desviacion: float) -> float
# retorna la probabilidad de x bajo una distribucion normal con media y desviacion dadas
def gaussiana(x, media, desviacion):
    return (1 / (sqrt(2 * pi) * desviacion)) * exp(-((x - media) ** 2) / (2 * desviacion ** 2))