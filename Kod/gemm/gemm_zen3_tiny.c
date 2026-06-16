#include "gemm.h"
#include "gemm_internal.h"
#include <immintrin.h>

#define ALWAYS_INLINE static inline __attribute__((always_inline))
#define HOT __attribute__((hot))

ALWAYS_INLINE HOT void ukr_4x12_tiny(int K,
                                     const double* __restrict A, int lda,
                                     const double* __restrict B, int ldb,
                                     double* __restrict C, int ldc,
                                     double alpha, double beta) {
    __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd(), c02 = _mm256_setzero_pd();
    __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd(), c12 = _mm256_setzero_pd();
    __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd(), c22 = _mm256_setzero_pd();
    __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd(), c32 = _mm256_setzero_pd();

    const double* a0 = A;
    const double* a1 = A + lda;
    const double* a2 = A + 2 * lda;
    const double* a3 = A + 3 * lda;

    for (int k = 0; k < K; k++) {
        __m256d b0 = _mm256_loadu_pd(&B[k * ldb + 0]);
        __m256d b1 = _mm256_loadu_pd(&B[k * ldb + 4]);
        __m256d b2 = _mm256_loadu_pd(&B[k * ldb + 8]);

        __m256d va = _mm256_broadcast_sd(&a0[k]);
        c00 = _mm256_fmadd_pd(va, b0, c00);
        c01 = _mm256_fmadd_pd(va, b1, c01);
        c02 = _mm256_fmadd_pd(va, b2, c02);

        va = _mm256_broadcast_sd(&a1[k]);
        c10 = _mm256_fmadd_pd(va, b0, c10);
        c11 = _mm256_fmadd_pd(va, b1, c11);
        c12 = _mm256_fmadd_pd(va, b2, c12);

        va = _mm256_broadcast_sd(&a2[k]);
        c20 = _mm256_fmadd_pd(va, b0, c20);
        c21 = _mm256_fmadd_pd(va, b1, c21);
        c22 = _mm256_fmadd_pd(va, b2, c22);

        va = _mm256_broadcast_sd(&a3[k]);
        c30 = _mm256_fmadd_pd(va, b0, c30);
        c31 = _mm256_fmadd_pd(va, b1, c31);
        c32 = _mm256_fmadd_pd(va, b2, c32);
    }

    __m256d av = _mm256_set1_pd(alpha);
    if (beta == 0.0) {
        /* Store-only epilog: no C load, no apply_beta pass needed beforehand. */
        _mm256_storeu_pd(&C[0*ldc+0], _mm256_mul_pd(av, c00));
        _mm256_storeu_pd(&C[0*ldc+4], _mm256_mul_pd(av, c01));
        _mm256_storeu_pd(&C[0*ldc+8], _mm256_mul_pd(av, c02));
        _mm256_storeu_pd(&C[1*ldc+0], _mm256_mul_pd(av, c10));
        _mm256_storeu_pd(&C[1*ldc+4], _mm256_mul_pd(av, c11));
        _mm256_storeu_pd(&C[1*ldc+8], _mm256_mul_pd(av, c12));
        _mm256_storeu_pd(&C[2*ldc+0], _mm256_mul_pd(av, c20));
        _mm256_storeu_pd(&C[2*ldc+4], _mm256_mul_pd(av, c21));
        _mm256_storeu_pd(&C[2*ldc+8], _mm256_mul_pd(av, c22));
        _mm256_storeu_pd(&C[3*ldc+0], _mm256_mul_pd(av, c30));
        _mm256_storeu_pd(&C[3*ldc+4], _mm256_mul_pd(av, c31));
        _mm256_storeu_pd(&C[3*ldc+8], _mm256_mul_pd(av, c32));
    } else {
        __m256d bv = _mm256_set1_pd(beta);
        _mm256_storeu_pd(&C[0*ldc+0], _mm256_fmadd_pd(av, c00, _mm256_mul_pd(bv, _mm256_loadu_pd(&C[0*ldc+0]))));
        _mm256_storeu_pd(&C[0*ldc+4], _mm256_fmadd_pd(av, c01, _mm256_mul_pd(bv, _mm256_loadu_pd(&C[0*ldc+4]))));
        _mm256_storeu_pd(&C[0*ldc+8], _mm256_fmadd_pd(av, c02, _mm256_mul_pd(bv, _mm256_loadu_pd(&C[0*ldc+8]))));
        _mm256_storeu_pd(&C[1*ldc+0], _mm256_fmadd_pd(av, c10, _mm256_mul_pd(bv, _mm256_loadu_pd(&C[1*ldc+0]))));
        _mm256_storeu_pd(&C[1*ldc+4], _mm256_fmadd_pd(av, c11, _mm256_mul_pd(bv, _mm256_loadu_pd(&C[1*ldc+4]))));
        _mm256_storeu_pd(&C[1*ldc+8], _mm256_fmadd_pd(av, c12, _mm256_mul_pd(bv, _mm256_loadu_pd(&C[1*ldc+8]))));
        _mm256_storeu_pd(&C[2*ldc+0], _mm256_fmadd_pd(av, c20, _mm256_mul_pd(bv, _mm256_loadu_pd(&C[2*ldc+0]))));
        _mm256_storeu_pd(&C[2*ldc+4], _mm256_fmadd_pd(av, c21, _mm256_mul_pd(bv, _mm256_loadu_pd(&C[2*ldc+4]))));
        _mm256_storeu_pd(&C[2*ldc+8], _mm256_fmadd_pd(av, c22, _mm256_mul_pd(bv, _mm256_loadu_pd(&C[2*ldc+8]))));
        _mm256_storeu_pd(&C[3*ldc+0], _mm256_fmadd_pd(av, c30, _mm256_mul_pd(bv, _mm256_loadu_pd(&C[3*ldc+0]))));
        _mm256_storeu_pd(&C[3*ldc+4], _mm256_fmadd_pd(av, c31, _mm256_mul_pd(bv, _mm256_loadu_pd(&C[3*ldc+4]))));
        _mm256_storeu_pd(&C[3*ldc+8], _mm256_fmadd_pd(av, c32, _mm256_mul_pd(bv, _mm256_loadu_pd(&C[3*ldc+8]))));
    }
}

ALWAYS_INLINE HOT void ukr_4x4_tiny(int K,
                                    const double* __restrict A, int lda,
                                    const double* __restrict B, int ldb,
                                    double* __restrict C, int ldc,
                                    double alpha, double beta) {
    __m256d c0 = _mm256_setzero_pd();
    __m256d c1 = _mm256_setzero_pd();
    __m256d c2 = _mm256_setzero_pd();
    __m256d c3 = _mm256_setzero_pd();

    const double* a0 = A;
    const double* a1 = A + lda;
    const double* a2 = A + 2 * lda;
    const double* a3 = A + 3 * lda;

    for (int k = 0; k < K; k++) {
        __m256d b = _mm256_loadu_pd(&B[k * ldb]);
        c0 = _mm256_fmadd_pd(_mm256_broadcast_sd(&a0[k]), b, c0);
        c1 = _mm256_fmadd_pd(_mm256_broadcast_sd(&a1[k]), b, c1);
        c2 = _mm256_fmadd_pd(_mm256_broadcast_sd(&a2[k]), b, c2);
        c3 = _mm256_fmadd_pd(_mm256_broadcast_sd(&a3[k]), b, c3);
    }

    __m256d av = _mm256_set1_pd(alpha);
    if (beta == 0.0) {
        _mm256_storeu_pd(&C[0 * ldc], _mm256_mul_pd(av, c0));
        _mm256_storeu_pd(&C[1 * ldc], _mm256_mul_pd(av, c1));
        _mm256_storeu_pd(&C[2 * ldc], _mm256_mul_pd(av, c2));
        _mm256_storeu_pd(&C[3 * ldc], _mm256_mul_pd(av, c3));
    } else {
        __m256d bv = _mm256_set1_pd(beta);
        _mm256_storeu_pd(&C[0*ldc], _mm256_fmadd_pd(av, c0, _mm256_mul_pd(bv, _mm256_loadu_pd(&C[0*ldc]))));
        _mm256_storeu_pd(&C[1*ldc], _mm256_fmadd_pd(av, c1, _mm256_mul_pd(bv, _mm256_loadu_pd(&C[1*ldc]))));
        _mm256_storeu_pd(&C[2*ldc], _mm256_fmadd_pd(av, c2, _mm256_mul_pd(bv, _mm256_loadu_pd(&C[2*ldc]))));
        _mm256_storeu_pd(&C[3*ldc], _mm256_fmadd_pd(av, c3, _mm256_mul_pd(bv, _mm256_loadu_pd(&C[3*ldc]))));
    }
}

static void scalar_edge(int K, int mr, int nr,
                        const double* A, int lda,
                        const double* B, int ldb,
                        double* C, int ldc,
                        double alpha, double beta) {
    for (int i = 0; i < mr; i++) {
        for (int j = 0; j < nr; j++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++)
                acc += A[i * lda + k] * B[k * ldb + j];
            double cv = (beta == 0.0) ? 0.0 : beta * C[i * ldc + j];
            C[i * ldc + j] = cv + alpha * acc;
        }
    }
}

void gemm_zen3_tiny(int M, int N, int K, double alpha,
                    const double* A, const double* B,
                    double beta, double* C) {
    int mb = (M / MR_Z) * MR_Z;
    int nb = (N / NR_Z) * NR_Z;

    for (int i = 0; i < mb; i += MR_Z) {
        for (int j = 0; j < nb; j += NR_Z) {
            ukr_4x12_tiny(K, &A[i * K], K, &B[j], N, &C[i * N + j], N, alpha, beta);
        }
        /* Vectorise the common N%12 = 4 and N%12 = 8 tails (e.g. N=64, N=256). */
        int j = nb;
        while (N - j >= 4) {
            ukr_4x4_tiny(K, &A[i * K], K, &B[j], N, &C[i * N + j], N, alpha, beta);
            j += 4;
        }
        if (j < N) {
            scalar_edge(K, MR_Z, N - j,
                        &A[i * K], K, &B[j], N,
                        &C[i * N + j], N, alpha, beta);
        }
    }
    if (mb < M) {
        scalar_edge(K, M - mb, N,
                    &A[mb * K], K, B, N,
                    &C[mb * N], N, alpha, beta);
    }
}
