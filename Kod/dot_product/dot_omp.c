#include "dot.h"
#include <omp.h>

extern double dot_simd_multiacc(const double* a, const double* b, size_t n);

double dot_omp(const double* a, const double* b, size_t n) {
    double total = 0.0;

    #pragma omp parallel reduction(+:total)
    {
        int nt   = omp_get_num_threads();
        int tid  = omp_get_thread_num();
        size_t chunk = n / nt;
        size_t start = (size_t)tid * chunk;
        size_t end   = (tid == nt - 1) ? n : start + chunk;
        total = dot_simd_multiacc(a + start, b + start, end - start);
    }
    return total;
}
