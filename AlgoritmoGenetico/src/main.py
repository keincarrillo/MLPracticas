import json
import sys
from math import log2
from modelo import (
    inicializar_poblacion, evaluar_poblacion,
    evolucionar, mejor_individuo, estadisticas_generacion
)
from utils import (
    fitness, cromosoma_a_gaps, proporcion_promedio,
    gaps_knuth, gaps_ciura, gaps_tokuda,
    gaps_shell_original, gaps_hibbard, gaps_sedgewick
)

# --- carga de configuracion ---
with open('../data/config.json', encoding='utf-8') as archivo:
    config = json.load(archivo)

n                = config['n_elementos']
tam_poblacion    = config['tam_poblacion']
num_generaciones = config['num_generaciones']
long_cromosoma   = config['long_cromosoma']
semilla          = config['semilla']

# -----------------------------------------------------------------------
# funciones de impresion
# -----------------------------------------------------------------------

ANCHO = 72

def separador(caracter="-"):
    print(caracter * ANCHO)

def titulo(texto):
    separador("=")
    print(f"  {texto}")
    separador("=")

def seccion(texto):
    print()
    separador()
    print(f"  {texto}")
    separador()

def tabla_fila(etiqueta, valor, ancho_etiqueta=38):
    print(f"  {etiqueta:<{ancho_etiqueta}}: {valor}")

# -----------------------------------------------------------------------
# reporte de configuracion
# -----------------------------------------------------------------------

def imprimir_configuracion():
    titulo("ALGORITMO GENETICO PARA SHELL SORT  —  Optimizacion de gaps")
    print()
    tabla_fila("Tamano del arreglo (n)", f"{n:.2e}  ({int(log2(n)):.0f} bits aprox)")
    tabla_fila("Tamano de poblacion",    tam_poblacion)
    tabla_fila("Numero de generaciones", num_generaciones)
    tabla_fila("Longitud del cromosoma", f"{long_cromosoma} genes (gaps)")
    tabla_fila("Tasa de mutacion",       f"{config['tasa_mutacion'] * 100:.0f} %")
    tabla_fila("Tasa de cruce",          f"{config['tasa_cruce'] * 100:.0f} %")
    tabla_fila("Tamano de torneo",       config['tam_torneo'])
    tabla_fila("Semilla aleatoria",      semilla if semilla is not None else "ninguna (aleatorio)")
    print()

# -----------------------------------------------------------------------
# evolucion con log por generacion
# -----------------------------------------------------------------------

def ejecutar_ag():
    poblacion = inicializar_poblacion(tam_poblacion, long_cromosoma, n, semilla)
    fitnesses = evaluar_poblacion(poblacion, n)

    mejor_global, fitness_global = mejor_individuo(poblacion, fitnesses)
    historial_fitness            = []
    historial_estadisticas       = []
    generacion_mejor             = 0

    seccion("PROGRESO DE EVOLUCION")
    encabezado = f"  {'Gen':>5}  {'Mejor fitness':>16}  {'Prom. poblacion':>16}  {'Desv. std':>14}  {'Gaps elite'}"
    print(encabezado)
    separador()

    for generacion in range(1, num_generaciones + 1):

        poblacion = evolucionar(poblacion, fitnesses, config, n)
        fitnesses = evaluar_poblacion(poblacion, n)
        stats     = estadisticas_generacion(fitnesses)

        candidato, fit_candidato = mejor_individuo(poblacion, fitnesses)

        if fit_candidato < fitness_global:
            fitness_global   = fit_candidato
            mejor_global     = candidato[:]
            generacion_mejor = generacion

        historial_fitness.append(fitness_global)
        historial_estadisticas.append(stats)

        # imprimir cada 25 generaciones, la primera y la ultima
        if generacion % 25 == 0 or generacion == 1 or generacion == num_generaciones:
            gaps_str = str(cromosoma_a_gaps(mejor_global))
            print(f"  {generacion:>5}  {fitness_global:>16.4e}  {stats['promedio']:>16.4e}  {stats['desviacion']:>14.4e}  {gaps_str}")

    return mejor_global, fitness_global, generacion_mejor, historial_fitness, historial_estadisticas

# -----------------------------------------------------------------------
# reporte del resultado del AG
# -----------------------------------------------------------------------

def imprimir_resultado_ag(mejor_global, fitness_global, generacion_mejor, historial_fitness):
    gaps_finales = cromosoma_a_gaps(mejor_global)

    seccion("RESULTADO DEL ALGORITMO GENETICO")
    tabla_fila("Mejor secuencia de gaps",        str(gaps_finales))
    tabla_fila("Numero de gaps",                 len(gaps_finales))
    tabla_fila("Gap mayor (primer paso)",        f"{gaps_finales[0]:.2e}")
    tabla_fila("Gap menor (ultimo paso)",        gaps_finales[-1])
    tabla_fila("Ratio promedio entre gaps",      f"{proporcion_promedio(gaps_finales):.4f}  (optimo teorico ~2.3)")
    tabla_fila("Fitness (comparaciones est.)",   f"{fitness_global:.6e}")
    tabla_fila("Generacion donde se encontro",  f"{generacion_mejor} de {num_generaciones}")
    tabla_fila("Mejora desde gen 1 a gen final", f"{((historial_fitness[0] - fitness_global) / historial_fitness[0] * 100):.2f} %")
    print()

    print("  Secuencia detallada:")
    separador()
    for i, g in enumerate(gaps_finales):
        ratio_str = ""
        if i < len(gaps_finales) - 1:
            ratio = g / gaps_finales[i + 1]
            ratio_str = f"  (ratio con siguiente: {ratio:.3f})"
        print(f"    h[{i + 1:>2}] = {g:>15,.0f}{ratio_str}")

# -----------------------------------------------------------------------
# comparacion con secuencias clasicas
# -----------------------------------------------------------------------

def imprimir_comparacion(fitness_ag, gaps_ag):
    secuencias = {
        "AG (este resultado)" : gaps_ag,
        "Shell original"      : gaps_shell_original(n),
        "Hibbard"             : gaps_hibbard(),
        "Knuth"               : gaps_knuth(n),
        "Sedgewick"           : gaps_sedgewick(),
        "Tokuda"              : gaps_tokuda(),
        "Ciura"               : gaps_ciura(),
    }

    seccion("COMPARACION CON SECUENCIAS CLASICAS")
    encabezado = f"  {'Secuencia':<20}  {'Num gaps':>8}  {'Fitness':>16}  {'vs AG':>12}  {'Ratio prom':>10}"
    print(encabezado)
    separador()

    resultados = []
    for nombre, gaps in secuencias.items():
        fit    = fitness(gaps, n)
        ratio  = proporcion_promedio(gaps)
        ngaps  = len(gaps)
        if nombre == "AG (este resultado)":
            diff_str = "  (referencia)"
        else:
            diff = ((fit - fitness_ag) / fitness_ag) * 100
            signo    = "+" if diff >= 0 else ""
            diff_str = f"  {signo}{diff:.2f} %"
        resultados.append((nombre, ngaps, fit, diff_str, ratio))

    # ordenar por fitness
    resultados.sort(key=lambda x: x[2])

    for nombre, ngaps, fit, diff_str, ratio in resultados:
        print(f"  {nombre:<20}  {ngaps:>8}  {fit:>16.4e}  {diff_str:>12}  {ratio:>10.3f}")

# -----------------------------------------------------------------------
# analisis del historial de evolucion
# -----------------------------------------------------------------------

def imprimir_analisis_historial(historial_fitness, historial_estadisticas):
    seccion("ANALISIS DEL HISTORIAL DE EVOLUCION")

    n_hist = len(historial_fitness)

    # calcular en que generaciones hubo mejoras significativas
    mejoras = []
    fit_anterior = historial_fitness[0]
    for i, f in enumerate(historial_fitness):
        if f < fit_anterior * 0.999:
            mejoras.append((i + 1, f, fit_anterior, (fit_anterior - f) / fit_anterior * 100))
            fit_anterior = f

    tabla_fila("Total de mejoras registradas",    len(mejoras))
    tabla_fila("Fitness generacion 1",            f"{historial_fitness[0]:.4e}")
    tabla_fila("Fitness generacion final",        f"{historial_fitness[-1]:.4e}")
    tabla_fila("Reduccion total del fitness",     f"{((historial_fitness[0] - historial_fitness[-1]) / historial_fitness[0] * 100):.4f} %")

    # convergencia: cuantas generaciones sin mejora al final
    sin_mejora = 0
    fit_ref    = historial_fitness[-1]
    for f in reversed(historial_fitness):
        if f == fit_ref:
            sin_mejora += 1
        else:
            break
    tabla_fila("Generaciones sin mejora al final", sin_mejora)
    tabla_fila("Convergencia alcanzada en gen",    n_hist - sin_mejora)
    print()

    if mejoras:
        print("  Mejoras significativas (>0.1% de reduccion):")
        separador()
        print(f"  {'Gen':>6}  {'Fitness nuevo':>16}  {'Fitness anterior':>16}  {'Mejora %':>10}")
        separador()
        for gen, nuevo, anterior, pct in mejoras[:20]:
            print(f"  {gen:>6}  {nuevo:>16.4e}  {anterior:>16.4e}  {pct:>9.4f}%")
        if len(mejoras) > 20:
            print(f"  ... y {len(mejoras) - 20} mejoras mas")

    # perfil de diversidad: desviacion promedio de la poblacion por tramos
    print()
    print("  Diversidad de la poblacion por tramos (desviacion estandar del fitness):")
    separador()
    tramo = max(1, n_hist // 10)
    print(f"  {'Tramo':<20}  {'Gen inicio':>10}  {'Gen fin':>10}  {'Desv. prom':>14}")
    separador()
    for i in range(0, n_hist, tramo):
        fin        = min(i + tramo, n_hist)
        desv_tramo = [s['desviacion'] for s in historial_estadisticas[i:fin]]
        desv_prom  = sum(desv_tramo) / len(desv_tramo)
        print(f"  {i // tramo + 1:<20}  {i + 1:>10}  {fin:>10}  {desv_prom:>14.4e}")

# -----------------------------------------------------------------------
# analisis teorico
# -----------------------------------------------------------------------

def imprimir_analisis_teorico(gaps_ag, fitness_ag):
    seccion("ANALISIS TEORICO DE LA SECUENCIA ENCONTRADA")

    gaps = cromosoma_a_gaps(gaps_ag)

    tabla_fila("n (tamano del arreglo)", f"{n:.4e}")
    tabla_fila("log2(n)",                f"{log2(n):.2f}")
    tabla_fila("sqrt(n)",                f"{n**0.5:.4e}")
    print()

    print("  Costo estimado por pasada (n * sqrt(h)):")
    separador()
    print(f"  {'Paso':>5}  {'Gap h':>15}  {'sqrt(h)':>12}  {'Comparaciones':>18}  {'% del total':>12}")
    separador()

    from math import sqrt
    costos = [(h, n * sqrt(h)) for h in sorted(gaps, reverse=True)]
    total  = sum(c for _, c in costos)

    for paso, (h, costo) in enumerate(costos, 1):
        pct = costo / total * 100
        print(f"  {paso:>5}  {h:>15,.0f}  {sqrt(h):>12.2f}  {costo:>18.4e}  {pct:>11.2f}%")

    separador()
    print(f"  {'TOTAL':>5}  {'':>15}  {'':>12}  {total:>18.4e}  {'100.00%':>12}")

    print()
    tabla_fila("Complejidad estimada total",       f"{fitness_ag:.4e} comparaciones")
    tabla_fila("Para referencia: n log2(n)",       f"{n * log2(n):.4e}  (QuickSort ideal)")
    tabla_fila("Factor vs QuickSort ideal",        f"{fitness_ag / (n * log2(n)):.4f} x")

# -----------------------------------------------------------------------
# resumen ejecutivo
# -----------------------------------------------------------------------

def imprimir_resumen(gaps_ag, fitness_ag):
    gaps = cromosoma_a_gaps(gaps_ag)

    seccion("RESUMEN EJECUTIVO")
    print(f"  La secuencia optima encontrada por el AG para n = {n:.2e} es:")
    print()
    print(f"  {gaps}")
    print()
    print(f"  Con {len(gaps)} gaps y un fitness de {fitness_ag:.4e} comparaciones estimadas.")
    print(f"  El ratio promedio entre gaps consecutivos es {proporcion_promedio(gaps):.3f}")
    print(f"  (el ratio teoricamente optimo ronda 2.2 - 2.5).")
    print()
    separador("=")

# -----------------------------------------------------------------------
# punto de entrada
# -----------------------------------------------------------------------

if __name__ == "__main__":

    imprimir_configuracion()

    mejor_global, fitness_global, generacion_mejor, historial_fitness, historial_stats = ejecutar_ag()

    gaps_finales = cromosoma_a_gaps(mejor_global)

    imprimir_resultado_ag(mejor_global, fitness_global, generacion_mejor, historial_fitness)
    imprimir_comparacion(fitness_global, gaps_finales)
    imprimir_analisis_historial(historial_fitness, historial_stats)
    imprimir_analisis_teorico(gaps_finales, fitness_global)
    imprimir_resumen(gaps_finales, fitness_global)