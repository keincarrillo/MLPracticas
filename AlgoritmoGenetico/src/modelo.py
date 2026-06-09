import random
from utils import fitness, cromosoma_a_gaps


# inicializar_poblacion(tam_poblacion: int, long_cromosoma: int, n: float, semilla: int | None) -> list[list[int]]
# genera una poblacion inicial aleatoria de cromosomas
# cada cromosoma es una lista de gaps enteros positivos, el ultimo siempre es 1
# el rango de inicializacion cubre desde gaps pequenos hasta gaps grandes (~30% de n)
def inicializar_poblacion(tam_poblacion, long_cromosoma, n, semilla=None):
    if semilla is not None:
        random.seed(semilla)

    limite_superior = int(n * 0.3)
    poblacion       = []

    for _ in range(tam_poblacion):
        cromosoma = [
            random.randint(1, max(2, limite_superior // (3 ** i)))
            for i in range(long_cromosoma)
        ]
        cromosoma[-1] = 1
        poblacion.append(cromosoma)

    return poblacion


# evaluar_poblacion(poblacion: list[list[int]], n: float) -> list[float]
# retorna el fitness de cada individuo en el mismo orden que la poblacion
def evaluar_poblacion(poblacion, n):
    return [fitness(cromosoma, n) for cromosoma in poblacion]


# seleccion_torneo(poblacion: list[list[int]], fitnesses: list[float], k: int) -> list[int]
# seleccion por torneo estocastico: elige k individuos al azar y retorna el de menor fitness
# k controla la presion selectiva — k mayor = mas presion, k=2 es el minimo util
# ventajas sobre ruleta: no sufre el problema de escala cuando hay un individuo
# mucho mejor que el resto, y mantiene mayor diversidad genetica en la poblacion
def seleccion_torneo(poblacion, fitnesses, k=3):
    candidatos = random.sample(range(len(poblacion)), min(k, len(poblacion)))
    ganador    = min(candidatos, key=lambda i: fitnesses[i])
    return poblacion[ganador]


# seleccion_sus(poblacion: list[list[int]], fitnesses: list[float]) -> list[int]
# seleccion universal estocastica (SUS): variante de ruleta con multiples punteros
# equidistantes en una sola tirada, eliminando el sesgo estadistico de ruleta simple
# se usa como alternativa cuando se quiere menor varianza en la seleccion
def seleccion_sus(poblacion, fitnesses):
    pares_validos = [(i, f) for i, f in enumerate(fitnesses) if f != float('inf')]
    if not pares_validos:
        return random.choice(poblacion)

    fitness_max = max(f for _, f in pares_validos)
    pesos       = [fitness_max - f for _, f in pares_validos]
    total       = sum(pesos)

    if total == 0:
        return poblacion[random.choice([i for i, _ in pares_validos])]

    punto     = random.uniform(0, total / len(pares_validos))
    acumulado = 0.0
    for indice, peso in zip([i for i, _ in pares_validos], pesos):
        acumulado += peso
        if acumulado >= punto:
            return poblacion[indice]

    return poblacion[pares_validos[-1][0]]


# cruce_un_punto(padre1: list[int], padre2: list[int]) -> tuple[list[int], list[int]]
# cruza dos padres en un punto aleatorio y retorna dos hijos
def cruce_un_punto(padre1, padre2):
    punto = random.randint(1, len(padre1) - 1)
    hijo1 = padre1[:punto] + padre2[punto:]
    hijo2 = padre2[:punto] + padre1[punto:]
    return hijo1, hijo2


# mutar(cromosoma: list[int], tasa_mutacion: float, n: float) -> list[int]
# aplica mutacion gen a gen con probabilidad tasa_mutacion
# desplaza cada gen un porcentaje aleatorio de su valor actual, respetando limites
def mutar(cromosoma, tasa_mutacion, n):
    limite_superior = int(n * 0.3)
    mutado          = cromosoma[:]

    for i in range(len(mutado) - 1):
        if random.random() < tasa_mutacion:
            desplazamiento = max(1, int(mutado[i] * random.uniform(0.1, 0.5)))
            direccion      = random.choice([-1, 1])
            nuevo_valor    = mutado[i] + direccion * desplazamiento
            mutado[i]      = max(1, min(nuevo_valor, limite_superior))

    mutado[-1] = 1
    return mutado


# evolucionar(poblacion, fitnesses, config, n) -> list[list[int]]
# genera la siguiente generacion aplicando elitismo, seleccion por torneo, cruce y mutacion
# usa seleccion_torneo con k=3 para mantener diversidad y evitar convergencia prematura
def evolucionar(poblacion, fitnesses, config, n):
    tam_poblacion = config['tam_poblacion']
    tasa_mutacion = config['tasa_mutacion']
    tasa_cruce    = config['tasa_cruce']
    k_torneo      = config.get('k_torneo', 3)

    # elitismo: el mejor individuo pasa sin cambios a la siguiente generacion
    indice_elite    = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
    nueva_poblacion = [poblacion[indice_elite][:]]

    while len(nueva_poblacion) < tam_poblacion:
        padre1 = seleccion_torneo(poblacion, fitnesses, k_torneo)
        padre2 = seleccion_torneo(poblacion, fitnesses, k_torneo)

        if random.random() < tasa_cruce:
            hijo1, hijo2 = cruce_un_punto(padre1, padre2)
        else:
            hijo1, hijo2 = padre1[:], padre2[:]

        hijo1 = mutar(hijo1, tasa_mutacion, n)
        hijo2 = mutar(hijo2, tasa_mutacion, n)

        nueva_poblacion.append(hijo1)
        if len(nueva_poblacion) < tam_poblacion:
            nueva_poblacion.append(hijo2)

    return nueva_poblacion


# mejor_individuo(poblacion: list[list[int]], fitnesses: list[float]) -> tuple[list[int], float]
# retorna el cromosoma con menor fitness y su valor de fitness
def mejor_individuo(poblacion, fitnesses):
    indice = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
    return poblacion[indice], fitnesses[indice]


# estadisticas_generacion(fitnesses: list[float]) -> dict
# calcula min, max, promedio y desviacion estandar del fitness de la generacion actual
def estadisticas_generacion(fitnesses):
    valores_validos = [f for f in fitnesses if f != float('inf')]
    if not valores_validos:
        return {'min': float('inf'), 'max': float('inf'), 'promedio': float('inf'), 'desviacion': 0.0}

    n          = len(valores_validos)
    minimo     = min(valores_validos)
    maximo     = max(valores_validos)
    promedio   = sum(valores_validos) / n
    varianza   = sum((f - promedio) ** 2 for f in valores_validos) / n
    from math import sqrt
    desviacion = sqrt(varianza)

    return {'min': minimo, 'max': maximo, 'promedio': promedio, 'desviacion': desviacion}