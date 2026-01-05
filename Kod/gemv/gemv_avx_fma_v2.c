#include "gemv.h"
#include <immintrin.h>
#include <string.h>
#include <stdint.h>

// ============================================================
// OpenBLAS-inspired GEMV implementation
// Key insight: minimize horizontal reductions by accumulating
// across entire row before reducing
// ============================================================

#define ALWAYS_INLINE static inline __attribute__((always_inline))
#define HOT __attribute__((hot))
#define ASSUME_ALIGNED(ptr, align) __builtin_assume_aligned(ptr, align)

// ============================================================
// Horizontal sum - called only once per row
// ============================================================
ALWAYS_INLINE double hsum_pd(__m256d v) {
    __m128d low  = _mm256_castpd256_pd128(v);
    __m128d high = _mm256_extractf128_pd(v, 1);
    low = _mm_add_pd(low, high);
    return _mm_cvtsd_f64(_mm_hadd_pd(low, low));
}

// ============================================================
// Core kernel: Process 4 rows at a time, entire column span
// Mimics OpenBLAS dgemv_t_microk_haswell-4.c structure
// ============================================================
ALWAYS_INLINE HOT void gemv_kernel_4rows(
    int cols,
    const double* __restrict a0,
    const double* __restrict a1,
    const double* __restrict a2,
    const double* __restrict a3,
    const double* __restrict x,
    double* __restrict y,
    double alpha)
{
    __m256d sum0 = _mm256_setzero_pd();
    __m256d sum1 = _mm256_setzero_pd();
    __m256d sum2 = _mm256_setzero_pd();
    __m256d sum3 = _mm256_setzero_pd();
    
    int j = 0;
    
    // Main loop: 8 elements per iteration (like OpenBLAS)
    #pragma GCC unroll 2
    for (; j + 7 < cols; j += 8) {
        __m256d x0 = _mm256_load_pd(&x[j]);
        __m256d x1 = _mm256_load_pd(&x[j + 4]);
        
        // Row 0
        sum0 = _mm256_fmadd_pd(_mm256_load_pd(&a0[j]), x0, sum0);
        sum0 = _mm256_fmadd_pd(_mm256_load_pd(&a0[j + 4]), x1, sum0);
        
        // Row 1
        sum1 = _mm256_fmadd_pd(_mm256_load_pd(&a1[j]), x0, sum1);
        sum1 = _mm256_fmadd_pd(_mm256_load_pd(&a1[j + 4]), x1, sum1);
        
        // Row 2
        sum2 = _mm256_fmadd_pd(_mm256_load_pd(&a2[j]), x0, sum2);
        sum2 = _mm256_fmadd_pd(_mm256_load_pd(&a2[j + 4]), x1, sum2);
        
        // Row 3
        sum3 = _mm256_fmadd_pd(_mm256_load_pd(&a3[j]), x0, sum3);
        sum3 = _mm256_fmadd_pd(_mm256_load_pd(&a3[j + 4]), x1, sum3);
    }
    
    // Handle 4 elements
    for (; j + 3 < cols; j += 4) {
        __m256d xv = _mm256_load_pd(&x[j]);
        sum0 = _mm256_fmadd_pd(_mm256_load_pd(&a0[j]), xv, sum0);
        sum1 = _mm256_fmadd_pd(_mm256_load_pd(&a1[j]), xv, sum1);
        sum2 = _mm256_fmadd_pd(_mm256_load_pd(&a2[j]), xv, sum2);
        sum3 = _mm256_fmadd_pd(_mm256_load_pd(&a3[j]), xv, sum3);
    }
    
    // Scalar tail
    double t0 = 0, t1 = 0, t2 = 0, t3 = 0;
    for (; j < cols; j++) {
        double xj = x[j];
        t0 += a0[j] * xj;
        t1 += a1[j] * xj;
        t2 += a2[j] * xj;
        t3 += a3[j] * xj;
    }
    
    // Single horizontal reduction per row (key optimization!)
    y[0] += alpha * (hsum_pd(sum0) + t0);
    y[1] += alpha * (hsum_pd(sum1) + t1);
    y[2] += alpha * (hsum_pd(sum2) + t2);
    y[3] += alpha * (hsum_pd(sum3) + t3);
}

// ============================================================
// Single row kernel for remainder
// ============================================================
ALWAYS_INLINE HOT void gemv_kernel_1row(
    int cols,
    const double* __restrict a,
    const double* __restrict x,
    double* __restrict y,
    double alpha)
{
    __m256d sum = _mm256_setzero_pd();
    
    int j = 0;
    for (; j + 7 < cols; j += 8) {
        __m256d x0 = _mm256_load_pd(&x[j]);
        __m256d x1 = _mm256_load_pd(&x[j + 4]);
        sum = _mm256_fmadd_pd(_mm256_load_pd(&a[j]), x0, sum);
        sum = _mm256_fmadd_pd(_mm256_load_pd(&a[j + 4]), x1, sum);
    }
    
    for (; j + 3 < cols; j += 4) {
        sum = _mm256_fmadd_pd(_mm256_load_pd(&a[j]), _mm256_load_pd(&x[j]), sum);
    }
    
    double t = 0;
    for (; j < cols; j++) {
        t += a[j] * x[j];
    }
    
    *y += alpha * (hsum_pd(sum) + t);
}

// ============================================================
// Main GEMV function - OpenBLAS style
// ============================================================
HOT void gemv_avx_fma_v2(int rows, int cols, double alpha,
                         const double* __restrict A,
                         const double* __restrict x,
                         double beta,
                         double* __restrict y)
{
    // Scale y by beta
    if (beta == 0.0) {
        memset(y, 0, (size_t)rows * sizeof(double));
    } else if (beta != 1.0) {
        for (int i = 0; i < rows; i++) {
            y[i] *= beta;
        }
    }
    
    const double* a_ptr = A;
    double* y_ptr = y;
    
    int i = 0;
    
    // Process 4 rows at a time
    for (; i + 3 < rows; i += 4) {
        gemv_kernel_4rows(cols,
                          a_ptr,
                          a_ptr + cols,
                          a_ptr + 2 * cols,
                          a_ptr + 3 * cols,
                          x, y_ptr, alpha);
        a_ptr += 4 * cols;
        y_ptr += 4;
    }
    
    // Handle remaining rows
    for (; i < rows; i++) {
        gemv_kernel_1row(cols, a_ptr, x, y_ptr, alpha);
        a_ptr += cols;
        y_ptr++;
    }
}
