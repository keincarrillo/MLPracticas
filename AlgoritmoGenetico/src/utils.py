from math import sqrt, log2

# comparaciones_shell(gaps: list[int], n: float) -> float
# estima el numero de comparaciones del Shell Sort para una secuencia de gaps y n elementos
# modelo teorico: cada pasada con gap h cuesta O(n * sqrt(h)) comparaciones
def comparaciones_shell(gaps, n):
    gaps_ordenados = sorted(set(gaps), reverse=True)
    if gaps_ordenados[-1] != 1:
        gaps_ordenados.append(1)
    total = 0.0
    for h in gaps_ordenados:
        total += n * sqrt(h)
    return total


# fitness(cromosoma: list[int], n: float) -> float
# calcula el costo de una secuencia de gaps (menor es mejor)
# penaliza gaps invalidos, ausencia de gap=1, y malas proporciones entre gaps consecutivos
def fitness(cromosoma, n):
    gaps = sorted(set(g for g in cromosoma if g >= 1), reverse=True)

    if not gaps:
        return float('inf')

    if any(g < 1 for g in gaps):
        return float('inf')

    if gaps[-1] != 1:
        gaps.append(1)

    if gaps[0] >= n:
        return float('inf')

    costo = comparaciones_shell(gaps, n)

    # penalizacion por proporciones entre gaps consecutivos
    # empiricamente el ratio optimo ronda 2.2-2.5
    penalizacion_ratio = 0.0
    for i in range(len(gaps) - 1):
        ratio = gaps[i] / gaps[i + 1]
        penalizacion_ratio += abs(ratio - 2.3) * (n * 0.01)

    return costo + penalizacion_ratio


# gaps_knuth(n: float) -> list[int]
# genera la secuencia de Knuth: h = 3*h + 1 mientras h < n/3
def gaps_knuth(n):
    gaps = []
    h = 1
    while h < n / 3:
        gaps.append(int(h))
        h = 3 * h + 1
    return sorted(gaps, reverse=True) if gaps else [1]


# gaps_ciura() -> list[int]
# retorna la secuencia empiricamente optima de Ciura (2001)
def gaps_ciura():
    return [701, 301, 132, 57, 23, 10, 4, 1]


# gaps_tokuda() -> list[int]
# genera la secuencia de Tokuda hasta un limite razonable
def gaps_tokuda():
    gaps = []
    k = 1
    while True:
        h = int((9 * (9 / 4) ** k - 4) / 5)
        if h > 10 ** 9:
            break
        gaps.append(h)
        k += 1
    return sorted(gaps, reverse=True) if gaps else [1]


# gaps_shell_original(n: float) -> list[int]
# genera la secuencia original de Shell: n/2, n/4, ..., 1
def gaps_shell_original(n):
    gaps = []
    h = int(n // 2)
    while h >= 1:
        gaps.append(h)
        h //= 2
    return gaps if gaps else [1]


# gaps_hibbard() -> list[int]
# genera la secuencia de Hibbard: 2^k - 1
def gaps_hibbard():
    gaps = []
    k = 1
    while True:
        h = (2 ** k) - 1
        if h > 10 ** 9:
            break
        gaps.append(h)
        k += 1
    return sorted(gaps, reverse=True) if gaps else [1]


# gaps_sedgewick() -> list[int]
# genera la secuencia de Sedgewick (1982)
def gaps_sedgewick():
    gaps = set()
    for k in range(0, 20):
        gaps.add(4 ** k + 3 * 2 ** (k - 1) + 1)
        gaps.add(2 ** (k + 2) * (2 ** (k + 2) - 3) + 1)
    return sorted([g for g in gaps if 1 <= g <= 10 ** 9], reverse=True)


# cromosoma_a_gaps(cromosoma: list[int]) -> list[int]
# convierte un cromosoma a secuencia de gaps ordenada de mayor a menor con 1 al final
def cromosoma_a_gaps(cromosoma):
    gaps = sorted(set(g for g in cromosoma if g >= 1), reverse=True)
    if not gaps or gaps[-1] != 1:
        gaps.append(1)
    return gaps


# proporcion_promedio(gaps: list[int]) -> float
# calcula el ratio promedio entre gaps consecutivos
def proporcion_promedio(gaps):
    gaps_ord = sorted(gaps, reverse=True)
    if len(gaps_ord) < 2:
        return 0.0
    ratios = [gaps_ord[i] / gaps_ord[i + 1] for i in range(len(gaps_ord) - 1)]
    return sum(ratios) / len(ratios)