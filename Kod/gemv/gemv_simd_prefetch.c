#include "gemv.h"
#include <immintrin.h>
#include <xmmintrin.h>
#include <string.h>

#define PREFETCH_DIST 8

void gemv_simd_prefetch(int rows, int cols, double alpha,
                        const double* A, const double* x,
                        double beta, double* y) {
    
    if (beta == 0.0) {
        memset(y, 0, rows * sizeof(double));
    } else if (beta != 1.0) {
        for (int i = 0; i < rows; i++) {
            y[i] *= beta;
        }
    }

    for (int i = 0; i < rows; i++) {
        __m256d sum0 = _mm256_setzero_pd();
        __m256d sum1 = _mm256_setzero_pd();
        
        if (i + PREFETCH_DIST < rows) {
            _mm_prefetch((const char*)&A[(i + PREFETCH_DIST) * cols], _MM_HINT_T0);
        }
        
        int j = 0;
        for (; j + 7 < cols; j += 8) {
            _mm_prefetch((const char*)&A[i * cols + j + 64], _MM_HINT_T0);
            
            __m256d a0 = _mm256_load_pd(&A[i * cols + j]);
            __m256d a1 = _mm256_load_pd(&A[i * cols + j + 4]);
            __m256d x0 = _mm256_load_pd(&x[j]);
            __m256d x1 = _mm256_load_pd(&x[j + 4]);
            
            sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(a0, x0));
            sum1 = _mm256_add_pd(sum1, _mm256_mul_pd(a1, x1));
        }
        
        sum0 = _mm256_add_pd(sum0, sum1);
        
        __m128d low = _mm256_castpd256_pd128(sum0);
        __m128d high = _mm256_extractf128_pd(sum0, 1);
        low = _mm_add_pd(low, high);
        low = _mm_hadd_pd(low, low);
        
        double sum = _mm_cvtsd_f64(low);
        
        for (; j < cols; j++) {
            sum += A[i * cols + j] * x[j];
        }
        
        y[i] += alpha * sum;
    }
}
