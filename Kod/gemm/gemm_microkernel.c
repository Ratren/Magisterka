#include "gemm.h"
#include "gemm_internal.h"
#include <immintrin.h>
#include <string.h>

#define ALWAYS_INLINE static inline __attribute__((always_inline))
#define HOT __attribute__((hot))

ALWAYS_INLINE HOT void ukr_6x8_unpacked(int K,
                                        const double* __restrict A, int lda,
                                        const double* __restrict B, int ldb,
                                        double* __restrict C, int ldc,
                                        double alpha) {
    __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd();
    __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd();
    __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd();
    __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd();
    __m256d c40 = _mm256_setzero_pd(), c41 = _mm256_setzero_pd();
    __m256d c50 = _mm256_setzero_pd(), c51 = _mm256_setzero_pd();

    const double* a0 = A + 0 * lda;
    const double* a1 = A + 1 * lda;
    const double* a2 = A + 2 * lda;
    const double* a3 = A + 3 * lda;
    const double* a4 = A + 4 * lda;
    const double* a5 = A + 5 * lda;

    for (int k = 0; k < K; k++) {
        __m256d b0 = _mm256_loadu_pd(&B[k * ldb + 0]);
        __m256d b1 = _mm256_loadu_pd(&B[k * ldb + 4]);

        __m256d va = _mm256_broadcast_sd(&a0[k]);
        c00 = _mm256_fmadd_pd(va, b0, c00);
        c01 = _mm256_fmadd_pd(va, b1, c01);
        va = _mm256_broadcast_sd(&a1[k]);
        c10 = _mm256_fmadd_pd(va, b0, c10);
        c11 = _mm256_fmadd_pd(va, b1, c11);
        va = _mm256_broadcast_sd(&a2[k]);
        c20 = _mm256_fmadd_pd(va, b0, c20);
        c21 = _mm256_fmadd_pd(va, b1, c21);
        va = _mm256_broadcast_sd(&a3[k]);
        c30 = _mm256_fmadd_pd(va, b0, c30);
        c31 = _mm256_fmadd_pd(va, b1, c31);
        va = _mm256_broadcast_sd(&a4[k]);
        c40 = _mm256_fmadd_pd(va, b0, c40);
        c41 = _mm256_fmadd_pd(va, b1, c41);
        va = _mm256_broadcast_sd(&a5[k]);
        c50 = _mm256_fmadd_pd(va, b0, c50);
        c51 = _mm256_fmadd_pd(va, b1, c51);
    }

    __m256d va = _mm256_set1_pd(alpha);
    _mm256_storeu_pd(&C[0 * ldc + 0], _mm256_fmadd_pd(va, c00, _mm256_loadu_pd(&C[0 * ldc + 0])));
    _mm256_storeu_pd(&C[0 * ldc + 4], _mm256_fmadd_pd(va, c01, _mm256_loadu_pd(&C[0 * ldc + 4])));
    _mm256_storeu_pd(&C[1 * ldc + 0], _mm256_fmadd_pd(va, c10, _mm256_loadu_pd(&C[1 * ldc + 0])));
    _mm256_storeu_pd(&C[1 * ldc + 4], _mm256_fmadd_pd(va, c11, _mm256_loadu_pd(&C[1 * ldc + 4])));
    _mm256_storeu_pd(&C[2 * ldc + 0], _mm256_fmadd_pd(va, c20, _mm256_loadu_pd(&C[2 * ldc + 0])));
    _mm256_storeu_pd(&C[2 * ldc + 4], _mm256_fmadd_pd(va, c21, _mm256_loadu_pd(&C[2 * ldc + 4])));
    _mm256_storeu_pd(&C[3 * ldc + 0], _mm256_fmadd_pd(va, c30, _mm256_loadu_pd(&C[3 * ldc + 0])));
    _mm256_storeu_pd(&C[3 * ldc + 4], _mm256_fmadd_pd(va, c31, _mm256_loadu_pd(&C[3 * ldc + 4])));
    _mm256_storeu_pd(&C[4 * ldc + 0], _mm256_fmadd_pd(va, c40, _mm256_loadu_pd(&C[4 * ldc + 0])));
    _mm256_storeu_pd(&C[4 * ldc + 4], _mm256_fmadd_pd(va, c41, _mm256_loadu_pd(&C[4 * ldc + 4])));
    _mm256_storeu_pd(&C[5 * ldc + 0], _mm256_fmadd_pd(va, c50, _mm256_loadu_pd(&C[5 * ldc + 0])));
    _mm256_storeu_pd(&C[5 * ldc + 4], _mm256_fmadd_pd(va, c51, _mm256_loadu_pd(&C[5 * ldc + 4])));
}

static void ukr_scalar_edge(int K, int mr, int nr,
                            const double* A, int lda,
                            const double* B, int ldb,
                            double* C, int ldc,
                            double alpha) {
    for (int i = 0; i < mr; i++) {
        for (int j = 0; j < nr; j++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++) {
                acc += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] += alpha * acc;
        }
    }
}

HOT void gemm_microkernel(int M, int N, int K, double alpha,
                          const double* A, const double* B,
                          double beta, double* C) {
    if (beta == 0.0) {
        for (int i = 0; i < M; i++)
            memset(&C[i * N], 0, N * sizeof(double));
    } else if (beta != 1.0) {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                C[i * N + j] *= beta;
    }

    int mb = (M / MR) * MR;
    int nb = (N / NR) * NR;

    for (int i = 0; i < mb; i += MR) {
        for (int j = 0; j < nb; j += NR) {
            ukr_6x8_unpacked(K, &A[i * K], K, &B[j], N, &C[i * N + j], N, alpha);
        }
        if (nb < N) {
            ukr_scalar_edge(K, MR, N - nb,
                            &A[i * K], K, &B[nb], N,
                            &C[i * N + nb], N, alpha);
        }
    }
    if (mb < M) {
        ukr_scalar_edge(K, M - mb, N,
                        &A[mb * K], K, B, N,
                        &C[mb * N], N, alpha);
    }
}
