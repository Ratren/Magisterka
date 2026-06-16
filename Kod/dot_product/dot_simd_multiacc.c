#include "dot.h"
#include <immintrin.h>

#define ALWAYS_INLINE static inline __attribute__((always_inline))
#define HOT __attribute__((hot))

ALWAYS_INLINE double hsum_pd(__m256d v) {
    __m128d low  = _mm256_castpd256_pd128(v);
    __m128d high = _mm256_extractf128_pd(v, 1);
    low = _mm_add_pd(low, high);
    return _mm_cvtsd_f64(_mm_hadd_pd(low, low));
}

HOT double dot_simd_multiacc(const double* __restrict a,
                             const double* __restrict b,
                             size_t n) {
    __m256d s0 = _mm256_setzero_pd();
    __m256d s1 = _mm256_setzero_pd();
    __m256d s2 = _mm256_setzero_pd();
    __m256d s3 = _mm256_setzero_pd();
    __m256d s4 = _mm256_setzero_pd();
    __m256d s5 = _mm256_setzero_pd();
    __m256d s6 = _mm256_setzero_pd();
    __m256d s7 = _mm256_setzero_pd();

    size_t i = 0;
    for (; i + 31 < n; i += 32) {
        s0 = _mm256_fmadd_pd(_mm256_loadu_pd(&a[i +  0]), _mm256_loadu_pd(&b[i +  0]), s0);
        s1 = _mm256_fmadd_pd(_mm256_loadu_pd(&a[i +  4]), _mm256_loadu_pd(&b[i +  4]), s1);
        s2 = _mm256_fmadd_pd(_mm256_loadu_pd(&a[i +  8]), _mm256_loadu_pd(&b[i +  8]), s2);
        s3 = _mm256_fmadd_pd(_mm256_loadu_pd(&a[i + 12]), _mm256_loadu_pd(&b[i + 12]), s3);
        s4 = _mm256_fmadd_pd(_mm256_loadu_pd(&a[i + 16]), _mm256_loadu_pd(&b[i + 16]), s4);
        s5 = _mm256_fmadd_pd(_mm256_loadu_pd(&a[i + 20]), _mm256_loadu_pd(&b[i + 20]), s5);
        s6 = _mm256_fmadd_pd(_mm256_loadu_pd(&a[i + 24]), _mm256_loadu_pd(&b[i + 24]), s6);
        s7 = _mm256_fmadd_pd(_mm256_loadu_pd(&a[i + 28]), _mm256_loadu_pd(&b[i + 28]), s7);
    }

    __m256d t0 = _mm256_add_pd(s0, s1);
    __m256d t1 = _mm256_add_pd(s2, s3);
    __m256d t2 = _mm256_add_pd(s4, s5);
    __m256d t3 = _mm256_add_pd(s6, s7);
    __m256d acc = _mm256_add_pd(_mm256_add_pd(t0, t1), _mm256_add_pd(t2, t3));

    for (; i + 3 < n; i += 4) {
        acc = _mm256_fmadd_pd(_mm256_loadu_pd(&a[i]), _mm256_loadu_pd(&b[i]), acc);
    }

    double sum = hsum_pd(acc);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}
