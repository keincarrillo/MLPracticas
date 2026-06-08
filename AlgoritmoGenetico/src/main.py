import json
import random
import sys
from math import log2
from modelo import (
    inicializar_poblacion, evaluar_poblacion,
    evolucionar, mejor_individuo, estadisticas_generacion
)
from utils import (
    fitness, cromosoma_a_gaps, proporcion_promedio,
    gaps_knuth, gaps_ciura, gaps_tokuda,
    gaps_shell_original, gaps_hibbard, gaps_sedgewick,
    shell_sort_real
)

# --- carga de configuracion ---
with open('../data/config.json', encoding='utf-8') as archivo:
    config = json.load(archivo)

# n para el AG (modelo teorico, puede ser grande)
n_teorico        = config['n_elementos']

# n real para ordenar (10^7 es manejable en RAM: ~80 MB con enteros Python)
N_REAL           = 10_000_000

tam_poblacion    = config['tam_poblacion']
num_generaciones = config['num_generaciones']
long_cromosoma   = config['long_cromosoma']
semilla          = config['semilla']

# -----------------------------------------------------------------------
# utilidades de impresion
# -----------------------------------------------------------------------

ANCHO = 76

def separador(c="-"):   print(c * ANCHO)
def titulo(texto):      separador("="); print(f"  {texto}"); separador("=")
def seccion(texto):     print(); separador(); print(f"  {texto}"); separador()

def tabla_fila(etiqueta, valor, ancho=40):
    print(f"  {etiqueta:<{ancho}}: {valor}")

# -----------------------------------------------------------------------
# configuracion
# -----------------------------------------------------------------------

# imprimir_configuracion() -> None
# imprime el encabezado con todos los parametros del experimento
def imprimir_configuracion():
    titulo("ALGORITMO GENETICO PARA SHELL SORT  —  Optimizacion de gaps")
    print()
    tabla_fila("n teorico (modelo fitness)",     f"{n_teorico:.2e}")
    tabla_fila("n real (arreglo a ordenar)",     f"{N_REAL:,}  (10^7 elementos)")
    tabla_fila("Memoria estimada del arreglo",   f"~{N_REAL * 28 / 1_048_576:.0f} MB  (enteros Python)")
    tabla_fila("Tamano de poblacion",            tam_poblacion)
    tabla_fila("Numero de generaciones",         num_generaciones)
    tabla_fila("Longitud del cromosoma",         f"{long_cromosoma} genes (gaps)")
    tabla_fila("Tasa de mutacion",               f"{config['tasa_mutacion'] * 100:.0f} %")
    tabla_fila("Tasa de cruce",                  f"{config['tasa_cruce'] * 100:.0f} %")
    tabla_fila("Semilla aleatoria",              semilla)
    print()

# -----------------------------------------------------------------------
# evolucion
# -----------------------------------------------------------------------

# ejecutar_ag() -> tuple[list[int], float, int, list[float], list[dict]]
# ejecuta el algoritmo genetico completo y retorna el mejor individuo, su fitness,
# la generacion donde se encontro, el historial de fitness y el historial de estadisticas
def ejecutar_ag():
    poblacion = inicializar_poblacion(tam_poblacion, long_cromosoma, n_teorico, semilla)
    fitnesses = evaluar_poblacion(poblacion, n_teorico)

    mejor_global, fitness_global = mejor_individuo(poblacion, fitnesses)
    historial_fitness      = []
    historial_estadisticas = []
    generacion_mejor       = 0

    seccion("PROGRESO DE EVOLUCION  (fitness teorico)")
    encabezado = f"  {'Gen':>5}  {'Mejor fitness':>16}  {'Prom. poblacion':>16}  {'Desv. std':>14}  {'Gaps elite'}"
    print(encabezado)
    separador()

    for gen in range(1, num_generaciones + 1):
        poblacion = evolucionar(poblacion, fitnesses, config, n_teorico)
        fitnesses = evaluar_poblacion(poblacion, n_teorico)
        stats     = estadisticas_generacion(fitnesses)
        cand, fc  = mejor_individuo(poblacion, fitnesses)

        if fc < fitness_global:
            fitness_global   = fc
            mejor_global     = cand[:]
            generacion_mejor = gen

        historial_fitness.append(fitness_global)
        historial_estadisticas.append(stats)

        if gen % 25 == 0 or gen == 1 or gen == num_generaciones:
            gaps_str = str(cromosoma_a_gaps(mejor_global))
            print(f"  {gen:>5}  {fitness_global:>16.4e}  {stats['promedio']:>16.4e}  {stats['desviacion']:>14.4e}  {gaps_str}")

    return mejor_global, fitness_global, generacion_mejor, historial_fitness, historial_estadisticas

# -----------------------------------------------------------------------
# generar arreglo real
# -----------------------------------------------------------------------

# generar_arreglo(semilla_arr: int | None) -> list[int]
# genera un arreglo aleatorio de N_REAL enteros con semilla reproducible
def generar_arreglo(semilla_arr=None):
    rng = random.Random(semilla_arr)
    return [rng.randint(0, N_REAL * 10) for _ in range(N_REAL)]

# -----------------------------------------------------------------------
# ordenamiento real con metricas completas
# -----------------------------------------------------------------------

# medir_ordenamiento_real(gaps: list[int], arreglo_base: list[int]) -> dict
# copia el arreglo base y lo ordena con shell_sort_real para no modificar el original
def medir_ordenamiento_real(gaps, arreglo_base):
    arr  = arreglo_base[:]
    return shell_sort_real(arr, gaps)

# imprimir_metricas_reales(nombre: str, gaps: list[int], metricas: dict) -> None
# imprime el detalle completo de comparaciones, intercambios, tiempo y memoria de un ordenamiento real
def imprimir_metricas_reales(nombre, gaps, metricas):
    seccion(f"ORDENAMIENTO REAL  —  {nombre}")
    tabla_fila("Gaps utilizados",           str(gaps))
    tabla_fila("Numero de gaps",            len(gaps))
    tabla_fila("Comparaciones reales",      f"{metricas['comparaciones']:,}")
    tabla_fila("Intercambios reales",       f"{metricas['intercambios']:,}")
    tabla_fila("Tiempo de ejecucion",       f"{metricas['tiempo_ms']:.2f} ms  ({metricas['tiempo_ms']/1000:.3f} s)")
    tabla_fila("Memoria pico (tracemalloc)",f"{metricas['memoria_kb']:.1f} KB")
    tabla_fila("Arreglo correctamente ord.",f"{'SI' if metricas['ordenado'] else 'NO  *** ERROR ***'}")
    tabla_fila("Comparaciones / n",         f"{metricas['comparaciones'] / N_REAL:.2f}")
    tabla_fila("Comparaciones / n·log2(n)", f"{metricas['comparaciones'] / (N_REAL * log2(N_REAL)):.4f}")
    print()
    print("  Detalle por pasada:")
    separador()
    print(f"  {'Paso':>4}  {'Gap':>12}  {'Comparaciones':>16}  {'Intercambios':>14}  {'% comps':>8}")
    separador()
    total_comp = metricas['comparaciones']
    for i, paso in enumerate(metricas['pasos'], 1):
        pct = paso['comparaciones'] / total_comp * 100 if total_comp > 0 else 0
        print(f"  {i:>4}  {paso['gap']:>12,}  {paso['comparaciones']:>16,}  {paso['intercambios']:>14,}  {pct:>7.2f}%")

# -----------------------------------------------------------------------
# comparacion real entre secuencias
# -----------------------------------------------------------------------

# comparacion_real_completa(gaps_ag: list[int], arreglo_base: list[int]) -> list[tuple]
# ordena el arreglo real con el AG y con todas las secuencias clasicas, imprime tabla comparativa
# retorna lista de tuplas (nombre, gaps, metricas) ordenada por comparaciones reales
def comparacion_real_completa(gaps_ag, arreglo_base):
    secuencias = {
        "AG (este resultado)" : gaps_ag,
        "Shell original"      : gaps_shell_original(N_REAL),
        "Hibbard"             : gaps_hibbard(),
        "Knuth"               : gaps_knuth(N_REAL),
        "Sedgewick"           : gaps_sedgewick(),
        "Tokuda"              : gaps_tokuda(),
        "Ciura"               : gaps_ciura(),
    }

    seccion("COMPARACION REAL ENTRE SECUENCIAS  (arreglo de 10^7 elementos)")
    print("  Ordenando con cada secuencia... (esto puede tardar varios minutos)")
    print()

    resultados = []
    for nombre, gaps in secuencias.items():
        sys.stdout.write(f"  Ordenando con {nombre:<22}...")
        sys.stdout.flush()
        m = medir_ordenamiento_real(gaps, arreglo_base)
        sys.stdout.write(f"  {m['tiempo_ms']:.0f} ms\n")
        sys.stdout.flush()
        resultados.append((nombre, gaps, m))

    # ordenar por comparaciones reales de menor a mayor
    resultados.sort(key=lambda x: x[2]['comparaciones'])

    print()
    separador()
    hdr = f"  {'Secuencia':<22}  {'Gaps':>5}  {'Comparaciones':>16}  {'Intercambios':>14}  {'Tiempo (ms)':>12}  {'vs AG':>10}"
    print(hdr)
    separador()

    comp_ag = next(m['comparaciones'] for n, _, m in resultados if n == "AG (este resultado)")

    for nombre, gaps, m in resultados:
        if nombre == "AG (este resultado)":
            diff_str = "referencia"
        else:
            diff = (m['comparaciones'] - comp_ag) / comp_ag * 100
            signo = "+" if diff >= 0 else ""
            diff_str = f"{signo}{diff:.1f}%"
        ok = "OK" if m['ordenado'] else "ERR"
        print(f"  {nombre:<22}  {len(gaps):>5}  {m['comparaciones']:>16,}  {m['intercambios']:>14,}  {m['tiempo_ms']:>11.0f}  {diff_str:>10}  {ok}")

    return resultados

# -----------------------------------------------------------------------
# resumen ejecutivo
# -----------------------------------------------------------------------

# imprimir_resumen_final(gaps_ag: list[int], metricas_ag: dict) -> None
# imprime el resumen ejecutivo con la secuencia optima y sus metricas reales
def imprimir_resumen_final(gaps_ag, metricas_ag):
    gaps = cromosoma_a_gaps(gaps_ag)
    seccion("RESUMEN EJECUTIVO")
    print(f"  Secuencia optima encontrada por el AG:")
    print(f"  {gaps}")
    print()
    print(f"  Ordenando {N_REAL:,} elementos aleatorios:")
    print(f"    - Comparaciones : {metricas_ag['comparaciones']:,}")
    print(f"    - Intercambios  : {metricas_ag['intercambios']:,}")
    print(f"    - Tiempo        : {metricas_ag['tiempo_ms']:.2f} ms")
    print(f"    - Ratio prom.   : {proporcion_promedio(gaps):.3f}  (optimo ~2.3)")
    print()
    separador("=")

# -----------------------------------------------------------------------
# punto de entrada
# -----------------------------------------------------------------------

if __name__ == "__main__":

    imprimir_configuracion()

    # --- fase 1: ejecutar el AG con modelo teorico para encontrar los gaps ---
    mejor_global, fitness_global, generacion_mejor, historial_fitness, historial_stats = ejecutar_ag()
    gaps_finales = cromosoma_a_gaps(mejor_global)

    print()
    seccion("MEJOR SECUENCIA ENCONTRADA POR EL AG")
    tabla_fila("Gaps", str(gaps_finales))
    tabla_fila("Num gaps", len(gaps_finales))
    tabla_fila("Fitness teorico", f"{fitness_global:.4e}")
    tabla_fila("Generacion", f"{generacion_mejor} de {num_generaciones}")
    tabla_fila("Ratio prom entre gaps", f"{proporcion_promedio(gaps_finales):.4f}")

    # --- fase 2: generar arreglo real de 10^7 elementos ---
    print()
    seccion("GENERANDO ARREGLO REAL DE 10^7 ELEMENTOS")
    import time
    t0 = time.perf_counter()
    print("  Generando arreglo aleatorio...", end=" ", flush=True)
    arreglo_base = generar_arreglo(semilla_arr=semilla)
    print(f"listo en {(time.perf_counter()-t0)*1000:.0f} ms")
    print(f"  Elementos: {len(arreglo_base):,}")
    print(f"  Rango de valores: [{min(arreglo_base[:1000])}, {max(arreglo_base[:1000])}] (muestra de 1000)")

    # --- fase 3: ordenamiento real con el AG y comparacion contra secuencias clasicas ---
    metricas_ag = medir_ordenamiento_real(gaps_finales, arreglo_base)
    imprimir_metricas_reales("AG (este resultado)", gaps_finales, metricas_ag)

    resultados = comparacion_real_completa(gaps_finales, arreglo_base)

    imprimir_resumen_final(gaps_finales, metricas_ag)