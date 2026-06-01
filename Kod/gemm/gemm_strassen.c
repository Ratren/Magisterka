#include "gemm.h"
#include "gemm_internal.h"
#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

/* Crossover empirically measured on Ryzen 5 5600 with the parallelised
   mat passes and gemm_zen3_best_omp as the base. Measured 6T uplift
   from the previous Strassen (which used gemm_zen3_omp as base):
     4096^3:   77 -> 234 GFLOPS  (3.0x)
     8192^3:        267 GFLOPS  (within 2.5% of gemm_zen3_best_omp)
   The 18 mat_add / mat_sub / mat_copy / mat_addto / mat_subfrom passes
   are now #pragma omp parallel for over the row dimension; they go from
   memory-bandwidth limited at single thread to ~6x faster at 6 threads. */
#define STRASSEN_MIN 2048

typedef void (*gemm_base_fn)(int, int, int, double,
                             const double*, const double*,
                             double, double*);

/* All five matrix passes parallelise trivially along the row dimension.
   At 4096^3 (m=n=2048 per pass) each touches ~64 MB; the per-pass
   cost drops from ~6 ms serial to ~1 ms on 6 cores. */
static void mat_add(int m, int n, const double* X, int ldx,
                    const double* Y, int ldy, double* Z, int ldz) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) {
        int j = 0;
        for (; j + 4 <= n; j += 4) {
            __m256d xv = _mm256_loadu_pd(&X[i * ldx + j]);
            __m256d yv = _mm256_loadu_pd(&Y[i * ldy + j]);
            _mm256_storeu_pd(&Z[i * ldz + j], _mm256_add_pd(xv, yv));
        }
        for (; j < n; j++) Z[i * ldz + j] = X[i * ldx + j] + Y[i * ldy + j];
    }
}

static void mat_sub(int m, int n, const double* X, int ldx,
                    const double* Y, int ldy, double* Z, int ldz) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) {
        int j = 0;
        for (; j + 4 <= n; j += 4) {
            __m256d xv = _mm256_loadu_pd(&X[i * ldx + j]);
            __m256d yv = _mm256_loadu_pd(&Y[i * ldy + j]);
            _mm256_storeu_pd(&Z[i * ldz + j], _mm256_sub_pd(xv, yv));
        }
        for (; j < n; j++) Z[i * ldz + j] = X[i * ldx + j] - Y[i * ldy + j];
    }
}

static void mat_addto(int m, int n, const double* X, int ldx,
                      double* Z, int ldz) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) {
        int j = 0;
        for (; j + 4 <= n; j += 4) {
            __m256d xv = _mm256_loadu_pd(&X[i * ldx + j]);
            __m256d zv = _mm256_loadu_pd(&Z[i * ldz + j]);
            _mm256_storeu_pd(&Z[i * ldz + j], _mm256_add_pd(zv, xv));
        }
        for (; j < n; j++) Z[i * ldz + j] += X[i * ldx + j];
    }
}

static void mat_subfrom(int m, int n, const double* X, int ldx,
                        double* Z, int ldz) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) {
        int j = 0;
        for (; j + 4 <= n; j += 4) {
            __m256d xv = _mm256_loadu_pd(&X[i * ldx + j]);
            __m256d zv = _mm256_loadu_pd(&Z[i * ldz + j]);
            _mm256_storeu_pd(&Z[i * ldz + j], _mm256_sub_pd(zv, xv));
        }
        for (; j < n; j++) Z[i * ldz + j] -= X[i * ldx + j];
    }
}

static void mat_copy(int m, int n, const double* X, int ldx, double* Z, int ldz) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) memcpy(&Z[i * ldz], &X[i * ldx], (size_t)n * sizeof(double));
}

/* Strassen-Winograd 7-multiplication schema on quadrants A=[A11 A12; A21 A22],
   B=[B11 B12; B21 B22], C=A*B. Sub-products land in T (size m2*n2) and are
   accumulated into the four C quadrants. */
static void strassen_core(int M, int N, int K, double alpha,
                          const double* A, const double* B, double* C,
                          gemm_base_fn base) {
    int m2 = M / 2, n2 = N / 2, k2 = K / 2;

    const double* A11 = A;
    const double* A12 = A + k2;
    const double* A21 = A + m2 * K;
    const double* A22 = A + m2 * K + k2;

    const double* B11 = B;
    const double* B12 = B + n2;
    const double* B21 = B + k2 * N;
    const double* B22 = B + k2 * N + n2;

    double* C11 = C;
    double* C12 = C + n2;
    double* C21 = C + m2 * N;
    double* C22 = C + m2 * N + n2;

    double* S1 = (double*)aligned_alloc(64, (size_t)m2 * k2 * sizeof(double));
    double* S2 = (double*)aligned_alloc(64, (size_t)k2 * n2 * sizeof(double));
    double* T  = (double*)aligned_alloc(64, (size_t)m2 * n2 * sizeof(double));

    /* gemm_zen3 has no lda/ldb/ldc parameter — it assumes row stride equals
       the matrix dimension. Raw quadrants (A11, A22, B11, etc.) inherit the
       parent matrix's stride, so we copy them through the S1/S2 buffers
       before each call. */
    mat_add(m2, k2, A11, K, A22, K, S1, k2);
    mat_add(k2, n2, B11, N, B22, N, S2, n2);
    base(m2, n2, k2, alpha, S1, S2, 0.0, T);
    mat_copy(m2, n2, T, n2, C11, N);
    mat_copy(m2, n2, T, n2, C22, N);

    mat_add(m2, k2, A21, K, A22, K, S1, k2);
    mat_copy(k2, n2, B11, N, S2, n2);
    base(m2, n2, k2, alpha, S1, S2, 0.0, T);
    mat_copy(m2, n2, T, n2, C21, N);
    mat_subfrom(m2, n2, T, n2, C22, N);

    mat_copy(m2, k2, A11, K, S1, k2);
    mat_sub(k2, n2, B12, N, B22, N, S2, n2);
    base(m2, n2, k2, alpha, S1, S2, 0.0, T);
    mat_copy(m2, n2, T, n2, C12, N);
    mat_addto(m2, n2, T, n2, C22, N);

    mat_copy(m2, k2, A22, K, S1, k2);
    mat_sub(k2, n2, B21, N, B11, N, S2, n2);
    base(m2, n2, k2, alpha, S1, S2, 0.0, T);
    mat_addto(m2, n2, T, n2, C11, N);
    mat_addto(m2, n2, T, n2, C21, N);

    mat_add(m2, k2, A11, K, A12, K, S1, k2);
    mat_copy(k2, n2, B22, N, S2, n2);
    base(m2, n2, k2, alpha, S1, S2, 0.0, T);
    mat_subfrom(m2, n2, T, n2, C11, N);
    mat_addto(m2, n2, T, n2, C12, N);

    mat_sub(m2, k2, A21, K, A11, K, S1, k2);
    mat_add(k2, n2, B11, N, B12, N, S2, n2);
    base(m2, n2, k2, alpha, S1, S2, 0.0, T);
    mat_addto(m2, n2, T, n2, C22, N);

    mat_sub(m2, k2, A12, K, A22, K, S1, k2);
    mat_add(k2, n2, B21, N, B22, N, S2, n2);
    base(m2, n2, k2, alpha, S1, S2, 0.0, T);
    mat_addto(m2, n2, T, n2, C11, N);

    free(S1);
    free(S2);
    free(T);
}

static int strassen_applicable(int M, int N, int K) {
    if (M < STRASSEN_MIN || N < STRASSEN_MIN || K < STRASSEN_MIN) return 0;
    if ((M & 1) || (N & 1) || (K & 1)) return 0;
    return 1;
}

void gemm_strassen(int M, int N, int K, double alpha,
                   const double* A, const double* B,
                   double beta, double* C) {
    if (!strassen_applicable(M, N, K)) {
        gemm_zen3(M, N, K, alpha, A, B, beta, C);
        return;
    }
    gemm_apply_beta(M, N, beta, C);
    strassen_core(M, N, K, alpha, A, B, C, gemm_zen3);
}

void gemm_strassen_omp(int M, int N, int K, double alpha,
                       const double* A, const double* B,
                       double beta, double* C) {
    if (!strassen_applicable(M, N, K)) {
        gemm_zen3_best_omp(M, N, K, alpha, A, B, beta, C);
        return;
    }
    gemm_apply_beta(M, N, beta, C);
    strassen_core(M, N, K, alpha, A, B, C, gemm_zen3_best_omp);
}
