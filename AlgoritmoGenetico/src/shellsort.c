#include <stdlib.h>
#include <string.h>

/*
 * shell_sort_c
 * arr       : arreglo de long long a ordenar (in-place)
 * n         : numero de elementos
 * gaps      : secuencia de gaps (de mayor a menor, debe terminar en 1)
 * num_gaps  : cantidad de gaps
 * out_comps : puntero donde se escribe el total de comparaciones
 * out_swaps : puntero donde se escribe el total de intercambios
 * out_comps_por_gap : arreglo pre-allocado de num_gaps longs para comparaciones por gap
 * out_swaps_por_gap : arreglo pre-allocado de num_gaps longs para intercambios por gap
 */
void shell_sort_c(
    long long *arr, long long n,
    long long *gaps, int num_gaps,
    long long *out_comps, long long *out_swaps,
    long long *out_comps_por_gap, long long *out_swaps_por_gap
) {
    long long total_comps = 0;
    long long total_swaps = 0;

    for (int g = 0; g < num_gaps; g++) {
        long long h = gaps[g];
        long long comp_pasada = 0;
        long long swap_pasada = 0;

        for (long long i = h; i < n; i++) {
            long long temp = arr[i];
            long long j = i;
            while (j >= h) {
                comp_pasada++;
                if (arr[j - h] > temp) {
                    arr[j] = arr[j - h];
                    j -= h;
                    swap_pasada++;
                } else {
                    break;
                }
            }
            arr[j] = temp;
        }

        out_comps_por_gap[g] = comp_pasada;
        out_swaps_por_gap[g] = swap_pasada;
        total_comps += comp_pasada;
        total_swaps += swap_pasada;
    }

    *out_comps = total_comps;
    *out_swaps = total_swaps;
}

/* verificar_ordenado: retorna 1 si arr esta ordenado, 0 si no */
int verificar_ordenado(long long *arr, long long n) {
    for (long long i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i + 1]) return 0;
    }
    return 1;
}