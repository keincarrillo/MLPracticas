import time
import tracemalloc
import ctypes
import os
import array
from math import sqrt, log2


# ---------------------------------------------------------------------------
# _cargar_lib() -> ctypes.CDLL
# carga la libreria compartida shellsort.so desde el mismo directorio que utils.py
# configura los tipos de argumento y retorno de las funciones exportadas
# lanza FileNotFoundError si el .so no existe (hay que compilarlo primero)
# ---------------------------------------------------------------------------

def _cargar_lib():
    directorio = os.path.dirname(os.path.abspath(__file__))
    ruta = os.path.join(directorio, 'shellsort.so')
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontro shellsort.so en {directorio}\n"
            f"Compila con:  gcc -O3 -march=native -shared -fPIC -o shellsort.so shellsort.c"
        )
    lib = ctypes.CDLL(ruta)

    lib.shell_sort_c.restype  = None
    lib.shell_sort_c.argtypes = [
        ctypes.POINTER(ctypes.c_longlong),  # arr
        ctypes.c_longlong,                  # n
        ctypes.POINTER(ctypes.c_longlong),  # gaps
        ctypes.c_int,                       # num_gaps
        ctypes.POINTER(ctypes.c_longlong),  # out_comps
        ctypes.POINTER(ctypes.c_longlong),  # out_swaps
        ctypes.POINTER(ctypes.c_longlong),  # out_comps_por_gap
        ctypes.POINTER(ctypes.c_longlong),  # out_swaps_por_gap
    ]

    lib.verificar_ordenado.restype  = ctypes.c_int
    lib.verificar_ordenado.argtypes = [
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.c_longlong,
    ]
    return lib


# _lib es un singleton: se carga una sola vez y se reutiliza en todas las llamadas
_lib = None


# _get_lib() -> ctypes.CDLL
# retorna la libreria C cargada, inicializandola la primera vez que se llama
def _get_lib():
    global _lib
    if _lib is None:
        _lib = _cargar_lib()
    return _lib


# _puntero(buf: array.array) -> ctypes.POINTER(ctypes.c_longlong)
# obtiene un puntero C directo al buffer interno de un array.array de tipo 'q'
# sin copiar los datos — equivalente a un cast de puntero en C
def _puntero(buf):
    addr, _ = buf.buffer_info()
    return ctypes.cast(addr, ctypes.POINTER(ctypes.c_longlong))


# comparaciones_shell(gaps: list[int], n: float) -> float
# estima el numero de comparaciones del Shell Sort para una secuencia de gaps y n elementos
# modelo corregido: normaliza el costo por numero de gaps para no favorecer secuencias cortas
# y penaliza fuertemente secuencias con poco cubrimiento inicial o menos de 4 gaps
def comparaciones_shell(gaps, n):
    gaps_ord = sorted(set(int(g) for g in gaps), reverse=True)
    if gaps_ord[-1] != 1:
        gaps_ord.append(1)
    k     = len(gaps_ord)
    total = sum((n * sqrt(h)) / sqrt(k) for h in gaps_ord)
    return total


# fitness(cromosoma: list[int], n: float) -> float
# calcula el costo de una secuencia de gaps (menor es mejor)
# penaliza: gaps invalidos, ausencia de gap=1, ratios entre gaps fuera de [2.0,3.0],
# cubrimiento inicial menor al 30% de n, y secuencias con menos de 4 gaps
def fitness(cromosoma, n):
    gaps = sorted(set(int(g) for g in cromosoma if g >= 1), reverse=True)

    if not gaps:
        return float('inf')

    if gaps[-1] != 1:
        gaps.append(1)

    if gaps[0] >= n:
        return float('inf')

    k    = len(gaps)
    cost = comparaciones_shell(gaps, n)

    # penalizacion cuadratica por ratios fuera del rango optimo
    penalizacion_ratio = 0.0
    for i in range(k - 1):
        ratio = gaps[i] / gaps[i + 1]
        penalizacion_ratio += abs(ratio - 2.3) ** 2 * n * 0.1

    # penalizacion por cubrimiento inicial insuficiente (gap maximo < 30% de n)
    cobertura = gaps[0] / n
    penalizacion_cobertura = max(0.0, (0.3 - cobertura) * n * 10)

    # penalizacion por secuencia demasiado corta (menos de 4 gaps)
    penalizacion_longitud = max(0, 4 - k) * n * 5

    return cost + penalizacion_ratio + penalizacion_cobertura + penalizacion_longitud


# shell_sort_real(arr: array.array, gaps: list[int]) -> dict
# ordena el arreglo arr en su lugar usando Shell Sort via libreria C compilada
# arr debe ser un array.array de tipo 'q' (long long) para evitar copias de memoria
# todos los gaps se convierten a int antes de crear el buffer para evitar TypeError
# retorna un diccionario con comparaciones, intercambios, tiempo_ms, memoria_kb,
# pasos (detalle por gap) y ordenado (verificacion de correctitud)
def shell_sort_real(arr, gaps):
    # forzar enteros para evitar TypeError con gaps que vengan como float
    gaps_ord = sorted(set(int(g) for g in gaps), reverse=True)
    if gaps_ord[-1] != 1:
        gaps_ord.append(1)

    lib      = _get_lib()
    n        = len(arr)
    num_gaps = len(gaps_ord)

    # gaps como buffer C sin copia, garantizando tipo int
    gaps_buf = array.array('q', gaps_ord)

    out_comps         = ctypes.c_longlong(0)
    out_swaps         = ctypes.c_longlong(0)
    out_comps_por_gap = (ctypes.c_longlong * num_gaps)()
    out_swaps_por_gap = (ctypes.c_longlong * num_gaps)()

    tracemalloc.start()
    inicio = time.perf_counter()

    lib.shell_sort_c(
        _puntero(arr), ctypes.c_longlong(n),
        _puntero(gaps_buf), ctypes.c_int(num_gaps),
        ctypes.byref(out_comps),
        ctypes.byref(out_swaps),
        out_comps_por_gap,
        out_swaps_por_gap,
    )

    fin = time.perf_counter()
    mem_actual, mem_pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    ordenado = bool(lib.verificar_ordenado(_puntero(arr), ctypes.c_longlong(n)))

    detalle_pasos = [
        {
            'gap'          : gaps_ord[g],
            'comparaciones': int(out_comps_por_gap[g]),
            'intercambios' : int(out_swaps_por_gap[g]),
        }
        for g in range(num_gaps)
    ]

    return {
        'comparaciones' : int(out_comps.value),
        'intercambios'  : int(out_swaps.value),
        'tiempo_ms'     : (fin - inicio) * 1000,
        'memoria_kb'    : mem_pico / 1024,
        'pasos'         : detalle_pasos,
        'ordenado'      : ordenado,
    }


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
# los valores se convierten a int explicitamente para evitar floats en el buffer C
def gaps_sedgewick():
    gaps = set()
    for k in range(0, 20):
        gaps.add(int(4 ** k + 3 * 2 ** (k - 1) + 1))
        gaps.add(int(2 ** (k + 2) * (2 ** (k + 2) - 3) + 1))
    return sorted([g for g in gaps if 1 <= g <= 10 ** 9], reverse=True)


# cromosoma_a_gaps(cromosoma: list[int]) -> list[int]
# convierte un cromosoma a secuencia de gaps ordenada de mayor a menor con 1 al final
def cromosoma_a_gaps(cromosoma):
    gaps = sorted(set(int(g) for g in cromosoma if g >= 1), reverse=True)
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