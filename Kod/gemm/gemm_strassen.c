#include "gemm.h"
#include "gemm_internal.h"
#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#define STRASSEN_MIN 2048

typedef void (*gemm_base_fn)(int, int, int, double,
                             const double*, const double*,
                             double, double*);

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
