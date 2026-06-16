#include "gemm.h"
#include "gemm_internal.h"
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

#define ALWAYS_INLINE static inline __attribute__((always_inline))
#define HOT __attribute__((hot))

void gemm_apply_beta(int M, int N, double beta, double* C) {
    if (beta == 0.0) {
        for (int i = 0; i < M; i++) memset(&C[i * N], 0, N * sizeof(double));
    } else if (beta != 1.0) {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) C[i * N + j] *= beta;
    }
}

void pack_A_panel(int mc, int kc, const double* A, int lda, double* A_pack) {
    int stripes = (mc + MR - 1) / MR;
    for (int s = 0; s < stripes; s++) {
        int i_base = s * MR;
        double* dst = A_pack + s * MR * kc;
        for (int k = 0; k < kc; k++) {
            for (int i = 0; i < MR; i++) {
                int row = i_base + i;
                dst[k * MR + i] = (row < mc) ? A[row * lda + k] : 0.0;
            }
        }
    }
}

void pack_B_panel(int kc, int nc, const double* B, int ldb, double* B_pack) {
    int stripes = (nc + NR - 1) / NR;
    for (int s = 0; s < stripes; s++) {
        int j_base = s * NR;
        double* dst = B_pack + s * NR * kc;
        for (int k = 0; k < kc; k++) {
            for (int j = 0; j < NR; j++) {
                int col = j_base + j;
                dst[k * NR + j] = (col < nc) ? B[k * ldb + col] : 0.0;
            }
        }
    }
}

ALWAYS_INLINE HOT void ukr_6x8_full(int kc,
                                    const double* __restrict A_pack,
                                    const double* __restrict B_pack,
                                    double* __restrict C, int ldc,
                                    double alpha) {
    __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd();
    __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd();
    __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd();
    __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd();
    __m256d c40 = _mm256_setzero_pd(), c41 = _mm256_setzero_pd();
    __m256d c50 = _mm256_setzero_pd(), c51 = _mm256_setzero_pd();

    for (int k = 0; k < kc; k++) {
        __m256d b0 = _mm256_load_pd(&B_pack[k * NR + 0]);
        __m256d b1 = _mm256_load_pd(&B_pack[k * NR + 4]);
        __m256d va;
        va = _mm256_broadcast_sd(&A_pack[k * MR + 0]);
        c00 = _mm256_fmadd_pd(va, b0, c00); c01 = _mm256_fmadd_pd(va, b1, c01);
        va = _mm256_broadcast_sd(&A_pack[k * MR + 1]);
        c10 = _mm256_fmadd_pd(va, b0, c10); c11 = _mm256_fmadd_pd(va, b1, c11);
        va = _mm256_broadcast_sd(&A_pack[k * MR + 2]);
        c20 = _mm256_fmadd_pd(va, b0, c20); c21 = _mm256_fmadd_pd(va, b1, c21);
        va = _mm256_broadcast_sd(&A_pack[k * MR + 3]);
        c30 = _mm256_fmadd_pd(va, b0, c30); c31 = _mm256_fmadd_pd(va, b1, c31);
        va = _mm256_broadcast_sd(&A_pack[k * MR + 4]);
        c40 = _mm256_fmadd_pd(va, b0, c40); c41 = _mm256_fmadd_pd(va, b1, c41);
        va = _mm256_broadcast_sd(&A_pack[k * MR + 5]);
        c50 = _mm256_fmadd_pd(va, b0, c50); c51 = _mm256_fmadd_pd(va, b1, c51);
    }

    __m256d alpha_v = _mm256_set1_pd(alpha);
    _mm256_storeu_pd(&C[0*ldc+0], _mm256_fmadd_pd(alpha_v, c00, _mm256_loadu_pd(&C[0*ldc+0])));
    _mm256_storeu_pd(&C[0*ldc+4], _mm256_fmadd_pd(alpha_v, c01, _mm256_loadu_pd(&C[0*ldc+4])));
    _mm256_storeu_pd(&C[1*ldc+0], _mm256_fmadd_pd(alpha_v, c10, _mm256_loadu_pd(&C[1*ldc+0])));
    _mm256_storeu_pd(&C[1*ldc+4], _mm256_fmadd_pd(alpha_v, c11, _mm256_loadu_pd(&C[1*ldc+4])));
    _mm256_storeu_pd(&C[2*ldc+0], _mm256_fmadd_pd(alpha_v, c20, _mm256_loadu_pd(&C[2*ldc+0])));
    _mm256_storeu_pd(&C[2*ldc+4], _mm256_fmadd_pd(alpha_v, c21, _mm256_loadu_pd(&C[2*ldc+4])));
    _mm256_storeu_pd(&C[3*ldc+0], _mm256_fmadd_pd(alpha_v, c30, _mm256_loadu_pd(&C[3*ldc+0])));
    _mm256_storeu_pd(&C[3*ldc+4], _mm256_fmadd_pd(alpha_v, c31, _mm256_loadu_pd(&C[3*ldc+4])));
    _mm256_storeu_pd(&C[4*ldc+0], _mm256_fmadd_pd(alpha_v, c40, _mm256_loadu_pd(&C[4*ldc+0])));
    _mm256_storeu_pd(&C[4*ldc+4], _mm256_fmadd_pd(alpha_v, c41, _mm256_loadu_pd(&C[4*ldc+4])));
    _mm256_storeu_pd(&C[5*ldc+0], _mm256_fmadd_pd(alpha_v, c50, _mm256_loadu_pd(&C[5*ldc+0])));
    _mm256_storeu_pd(&C[5*ldc+4], _mm256_fmadd_pd(alpha_v, c51, _mm256_loadu_pd(&C[5*ldc+4])));
}

static void ukr_6x8_edge(int kc, int mr, int nr,
                         const double* A_pack, const double* B_pack,
                         double* C, int ldc, double alpha) {
    for (int i = 0; i < mr; i++) {
        for (int j = 0; j < nr; j++) {
            double acc = 0.0;
            for (int k = 0; k < kc; k++) {
                acc += A_pack[k * MR + i] * B_pack[k * NR + j];
            }
            C[i * ldc + j] += alpha * acc;
        }
    }
}

void macrokernel(int mc, int nc, int kc,
                 const double* A_pack, const double* B_pack,
                 double* C, int ldc) {
    for (int ir = 0; ir < mc; ir += MR) {
        int mr = gemm_imin(MR, mc - ir);
        const double* A_stripe = A_pack + (ir / MR) * MR * kc;
        for (int jr = 0; jr < nc; jr += NR) {
            int nr = gemm_imin(NR, nc - jr);
            const double* B_stripe = B_pack + (jr / NR) * NR * kc;
            double* C_tile = C + ir * ldc + jr;
            if (mr == MR && nr == NR) {
                ukr_6x8_full(kc, A_stripe, B_stripe, C_tile, ldc, 1.0);
            } else {
                ukr_6x8_edge(kc, mr, nr, A_stripe, B_stripe, C_tile, ldc, 1.0);
            }
        }
    }
}

void gemm_packed_core(int M, int N, int K, double alpha,
                      const double* A, const double* B, double* C,
                      double* A_pack, double* B_pack) {
    for (int jc = 0; jc < N; jc += NC) {
        int nc = gemm_imin(NC, N - jc);
        for (int pc = 0; pc < K; pc += KC) {
            int kc = gemm_imin(KC, K - pc);
            pack_B_panel(kc, nc, B + pc * N + jc, N, B_pack);
            for (int ic = 0; ic < M; ic += MC) {
                int mc = gemm_imin(MC, M - ic);
                pack_A_panel(mc, kc, A + ic * K + pc, K, A_pack);

                if (alpha == 1.0) {
                    macrokernel(mc, nc, kc, A_pack, B_pack, C + ic * N + jc, N);
                } else {
                    for (int ir = 0; ir < mc; ir += MR) {
                        int mr = gemm_imin(MR, mc - ir);
                        const double* A_stripe = A_pack + (ir / MR) * MR * kc;
                        for (int jr = 0; jr < nc; jr += NR) {
                            int nr = gemm_imin(NR, nc - jr);
                            const double* B_stripe = B_pack + (jr / NR) * NR * kc;
                            double* C_tile = C + (ic + ir) * N + jc + jr;
                            if (mr == MR && nr == NR)
                                ukr_6x8_full(kc, A_stripe, B_stripe, C_tile, N, alpha);
                            else
                                ukr_6x8_edge(kc, mr, nr, A_stripe, B_stripe, C_tile, N, alpha);
                        }
                    }
                }
            }
        }
    }
}

void gemm_packed(int M, int N, int K, double alpha,
                 const double* A, const double* B,
                 double beta, double* C) {
    gemm_apply_beta(M, N, beta, C);

    int mb_eff = gemm_round_up(gemm_imin(MC, M), MR);
    int nb_eff = gemm_round_up(gemm_imin(NC, N), NR);
    int kc_eff = gemm_imin(KC, K);

    double* A_pack = (double*)aligned_alloc(64, (size_t)mb_eff * kc_eff * sizeof(double));
    double* B_pack = (double*)aligned_alloc(64, (size_t)kc_eff * nb_eff * sizeof(double));

    gemm_packed_core(M, N, K, alpha, A, B, C, A_pack, B_pack);

    free(A_pack);
    free(B_pack);
}
