#include "dot.h"
#include <immintrin.h>

static inline double hsum_pd(__m256d v) {
    __m128d low  = _mm256_castpd256_pd128(v);
    __m128d high = _mm256_extractf128_pd(v, 1);
    low = _mm_add_pd(low, high);
    return _mm_cvtsd_f64(_mm_hadd_pd(low, low));
}

double dot_simd(const double* a, const double* b, size_t n) {
    __m256d acc = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        acc = _mm256_fmadd_pd(va, vb, acc);
    }
    double sum = hsum_pd(acc);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}
