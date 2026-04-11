#include "gemv.h"
#include <immintrin.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>

// V3 OpenMP: Parallel GEMV for Zen 3
//
// - Each thread gets a contiguous block of rows (aligned to 4)
// - No false sharing: each thread writes separate y region
// - x vector shared read-only (fits in L3, broadcast efficient)

#define ALWAYS_INLINE static inline __attribute__((always_inline))
#define HOT __attribute__((hot))

// Don't parallelize if too few rows or elements 
#define MIN_ROWS_FOR_PARALLEL 128
#define MIN_ELEMENTS_FOR_PARALLEL (256 * 256)
// Very large matrices are memory-bound; multi-threaded access causes
// contention on the memory controller and hurts performance.
// Threshold at 8M covers medium (1M) but skips large (16M).
#define MAX_ELEMENTS_FOR_PARALLEL (8 * 1024 * 1024)

ALWAYS_INLINE double hsum_pd(__m256d v) {
    __m128d low  = _mm256_castpd256_pd128(v);
    __m128d high = _mm256_extractf128_pd(v, 1);
    low = _mm_add_pd(low, high);
    return _mm_cvtsd_f64(_mm_hadd_pd(low, low));
}

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
    __m256d sum0a = _mm256_setzero_pd();
    __m256d sum0b = _mm256_setzero_pd();
    __m256d sum1a = _mm256_setzero_pd();
    __m256d sum1b = _mm256_setzero_pd();
    __m256d sum2a = _mm256_setzero_pd();
    __m256d sum2b = _mm256_setzero_pd();
    __m256d sum3a = _mm256_setzero_pd();
    __m256d sum3b = _mm256_setzero_pd();

    int j = 0;

    for (; j + 15 < cols; j += 16) {
        __m256d x0 = _mm256_load_pd(&x[j]);
        __m256d x1 = _mm256_load_pd(&x[j + 4]);
        __m256d x2 = _mm256_load_pd(&x[j + 8]);
        __m256d x3 = _mm256_load_pd(&x[j + 12]);

        sum0a = _mm256_fmadd_pd(_mm256_load_pd(&a0[j]),      x0, sum0a);
        sum0b = _mm256_fmadd_pd(_mm256_load_pd(&a0[j + 4]),  x1, sum0b);
        sum0a = _mm256_fmadd_pd(_mm256_load_pd(&a0[j + 8]),  x2, sum0a);
        sum0b = _mm256_fmadd_pd(_mm256_load_pd(&a0[j + 12]), x3, sum0b);

        sum1a = _mm256_fmadd_pd(_mm256_load_pd(&a1[j]),      x0, sum1a);
        sum1b = _mm256_fmadd_pd(_mm256_load_pd(&a1[j + 4]),  x1, sum1b);
        sum1a = _mm256_fmadd_pd(_mm256_load_pd(&a1[j + 8]),  x2, sum1a);
        sum1b = _mm256_fmadd_pd(_mm256_load_pd(&a1[j + 12]), x3, sum1b);

        sum2a = _mm256_fmadd_pd(_mm256_load_pd(&a2[j]),      x0, sum2a);
        sum2b = _mm256_fmadd_pd(_mm256_load_pd(&a2[j + 4]),  x1, sum2b);
        sum2a = _mm256_fmadd_pd(_mm256_load_pd(&a2[j + 8]),  x2, sum2a);
        sum2b = _mm256_fmadd_pd(_mm256_load_pd(&a2[j + 12]), x3, sum2b);

        sum3a = _mm256_fmadd_pd(_mm256_load_pd(&a3[j]),      x0, sum3a);
        sum3b = _mm256_fmadd_pd(_mm256_load_pd(&a3[j + 4]),  x1, sum3b);
        sum3a = _mm256_fmadd_pd(_mm256_load_pd(&a3[j + 8]),  x2, sum3a);
        sum3b = _mm256_fmadd_pd(_mm256_load_pd(&a3[j + 12]), x3, sum3b);
    }

    __m256d sum0 = _mm256_add_pd(sum0a, sum0b);
    __m256d sum1 = _mm256_add_pd(sum1a, sum1b);
    __m256d sum2 = _mm256_add_pd(sum2a, sum2b);
    __m256d sum3 = _mm256_add_pd(sum3a, sum3b);

    for (; j + 7 < cols; j += 8) {
        __m256d x0 = _mm256_load_pd(&x[j]);
        __m256d x1 = _mm256_load_pd(&x[j + 4]);

        sum0 = _mm256_fmadd_pd(_mm256_load_pd(&a0[j]),     x0, sum0);
        sum0 = _mm256_fmadd_pd(_mm256_load_pd(&a0[j + 4]), x1, sum0);
        sum1 = _mm256_fmadd_pd(_mm256_load_pd(&a1[j]),     x0, sum1);
        sum1 = _mm256_fmadd_pd(_mm256_load_pd(&a1[j + 4]), x1, sum1);
        sum2 = _mm256_fmadd_pd(_mm256_load_pd(&a2[j]),     x0, sum2);
        sum2 = _mm256_fmadd_pd(_mm256_load_pd(&a2[j + 4]), x1, sum2);
        sum3 = _mm256_fmadd_pd(_mm256_load_pd(&a3[j]),     x0, sum3);
        sum3 = _mm256_fmadd_pd(_mm256_load_pd(&a3[j + 4]), x1, sum3);
    }

    for (; j + 3 < cols; j += 4) {
        __m256d xv = _mm256_load_pd(&x[j]);
        sum0 = _mm256_fmadd_pd(_mm256_load_pd(&a0[j]), xv, sum0);
        sum1 = _mm256_fmadd_pd(_mm256_load_pd(&a1[j]), xv, sum1);
        sum2 = _mm256_fmadd_pd(_mm256_load_pd(&a2[j]), xv, sum2);
        sum3 = _mm256_fmadd_pd(_mm256_load_pd(&a3[j]), xv, sum3);
    }

    double t0 = 0, t1 = 0, t2 = 0, t3 = 0;
    for (; j < cols; j++) {
        double xj = x[j];
        t0 += a0[j] * xj;
        t1 += a1[j] * xj;
        t2 += a2[j] * xj;
        t3 += a3[j] * xj;
    }

    y[0] += alpha * (hsum_pd(sum0) + t0);
    y[1] += alpha * (hsum_pd(sum1) + t1);
    y[2] += alpha * (hsum_pd(sum2) + t2);
    y[3] += alpha * (hsum_pd(sum3) + t3);
}

ALWAYS_INLINE HOT void gemv_kernel_1row(
    int cols,
    const double* __restrict a,
    const double* __restrict x,
    double* __restrict y,
    double alpha)
{
    __m256d sum = _mm256_setzero_pd();
    __m256d sum2 = _mm256_setzero_pd();

    int j = 0;
    for (; j + 7 < cols; j += 8) {
        __m256d x0 = _mm256_load_pd(&x[j]);
        __m256d x1 = _mm256_load_pd(&x[j + 4]);
        sum  = _mm256_fmadd_pd(_mm256_load_pd(&a[j]),     x0, sum);
        sum2 = _mm256_fmadd_pd(_mm256_load_pd(&a[j + 4]), x1, sum2);
    }

    sum = _mm256_add_pd(sum, sum2);

    for (; j + 3 < cols; j += 4) {
        sum = _mm256_fmadd_pd(_mm256_load_pd(&a[j]), _mm256_load_pd(&x[j]), sum);
    }

    double t = 0;
    for (; j < cols; j++) {
        t += a[j] * x[j];
    }

    *y += alpha * (hsum_pd(sum) + t);
}

static inline void process_row_range(
    int start_row, int end_row, int cols,
    double alpha,
    const double* __restrict A,
    const double* __restrict x,
    double* __restrict y)
{
    const double* a_ptr = A + (size_t)start_row * cols;
    double* y_ptr = y + start_row;

    int i = start_row;

    // Process 4 rows at a time
    for (; i + 3 < end_row; i += 4) {
        gemv_kernel_4rows(cols,
                          a_ptr,
                          a_ptr + cols,
                          a_ptr + 2 * cols,
                          a_ptr + 3 * cols,
                          x, y_ptr, alpha);
        a_ptr += 4 * cols;
        y_ptr += 4;
    }

    // Handle remaining 1-3 rows
    for (; i < end_row; i++) {
        gemv_kernel_1row(cols, a_ptr, x, y_ptr, alpha);
        a_ptr += cols;
        y_ptr++;
    }
}

// Uses a single omp parallel region with manual row division
// instead of parallel for — avoids per-iteration OMP overhead.
// Thread boundaries aligned to 4-row groups for SIMD efficiency.
HOT void gemv_avx_fma_v3_omp(int rows, int cols, double alpha,
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

    size_t total_elements = (size_t)rows * cols;

    // too small (thread overhead) or too large (memory-bound)
    if (rows < MIN_ROWS_FOR_PARALLEL ||
        total_elements < MIN_ELEMENTS_FOR_PARALLEL ||
        total_elements > MAX_ELEMENTS_FOR_PARALLEL) {
        process_row_range(0, rows, cols, alpha, A, x, y);
        return;
    }

    #pragma omp parallel
    {
        int nt = omp_get_num_threads();
        int tid = omp_get_thread_num();

        // Divide rows into 4-row chunks, distribute evenly
        int total_chunks = (rows + 3) / 4;
        int chunks_per_thread = total_chunks / nt;
        int extra = total_chunks % nt;

        // First 'extra' threads get one additional chunk
        int start_chunk = tid * chunks_per_thread + (tid < extra ? tid : extra);
        int end_chunk = start_chunk + chunks_per_thread + (tid < extra ? 1 : 0);

        int start_row = start_chunk * 4;
        int end_row = end_chunk * 4;
        if (end_row > rows) end_row = rows;

        if (start_row < rows) {
            process_row_range(start_row, end_row, cols, alpha, A, x, y);
        }
    }
}
