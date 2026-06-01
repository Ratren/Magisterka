#include "gemm.h"
#include "gemm_internal.h"
#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>

#define ALWAYS_INLINE static inline __attribute__((always_inline))
#define HOT __attribute__((hot))

/* No software prefetch inside the kernel: with 12 accumulators + 3 B regs +
   1 A reg the YMM file is full, and the prefetch's address-calc plus branch
   adds enough live values to push accumulators to the stack. Agner Fog's
   microarchitecture manual §23.16 explicitly recommends relying on Zen 3's
   HW prefetcher for streaming reads. */

void pack_A_panel_z(int mc, int kc, const double* A, int lda, double* A_pack) {
    int stripes = (mc + MR_Z - 1) / MR_Z;
    for (int s = 0; s < stripes; s++) {
        int i_base = s * MR_Z;
        double* dst = A_pack + s * MR_Z * kc;
        for (int k = 0; k < kc; k++) {
            for (int i = 0; i < MR_Z; i++) {
                int row = i_base + i;
                dst[k * MR_Z + i] = (row < mc) ? A[row * lda + k] : 0.0;
            }
        }
    }
}

void pack_B_panel_z(int kc, int nc, const double* B, int ldb, double* B_pack) {
    int stripes = (nc + NR_Z - 1) / NR_Z;
    for (int s = 0; s < stripes; s++) {
        int j_base = s * NR_Z;
        double* dst = B_pack + s * NR_Z * kc;
        for (int k = 0; k < kc; k++) {
            for (int j = 0; j < NR_Z; j++) {
                int col = j_base + j;
                dst[k * NR_Z + j] = (col < nc) ? B[k * ldb + col] : 0.0;
            }
        }
    }
}

/* The K loop body is written in inline assembly so we can pin all 12
   accumulators to specific YMMs and prevent the spilling GCC otherwise does
   when the register file is fully saturated (12 acc + 1 A broadcast + 3 B
   = 16 YMMs, leaving 0 spare). YMM allocation: ymm4..ymm15 = 12 accs,
   ymm0 = A broadcast, ymm1..ymm3 = B vectors. */
ALWAYS_INLINE HOT void ukr_4x12_full(int kc,
                                     const double* __restrict A_pack,
                                     const double* __restrict B_pack,
                                     double* __restrict C, int ldc,
                                     double alpha) {
    long k_ctr = kc;
    __asm__ volatile (
        "vxorpd %%ymm4,  %%ymm4,  %%ymm4  \n"
        "vxorpd %%ymm5,  %%ymm5,  %%ymm5  \n"
        "vxorpd %%ymm6,  %%ymm6,  %%ymm6  \n"
        "vxorpd %%ymm7,  %%ymm7,  %%ymm7  \n"
        "vxorpd %%ymm8,  %%ymm8,  %%ymm8  \n"
        "vxorpd %%ymm9,  %%ymm9,  %%ymm9  \n"
        "vxorpd %%ymm10, %%ymm10, %%ymm10 \n"
        "vxorpd %%ymm11, %%ymm11, %%ymm11 \n"
        "vxorpd %%ymm12, %%ymm12, %%ymm12 \n"
        "vxorpd %%ymm13, %%ymm13, %%ymm13 \n"
        "vxorpd %%ymm14, %%ymm14, %%ymm14 \n"
        "vxorpd %%ymm15, %%ymm15, %%ymm15 \n"

        ".balign 32                                  \n"
        "1:                                          \n"
        "vmovapd  0(%[B]),  %%ymm1                   \n"
        "vmovapd  32(%[B]), %%ymm2                   \n"
        "vmovapd  64(%[B]), %%ymm3                   \n"
        "vbroadcastsd 0(%[A]), %%ymm0                \n"
        "vfmadd231pd %%ymm1, %%ymm0, %%ymm4          \n"
        "vfmadd231pd %%ymm2, %%ymm0, %%ymm5          \n"
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm6          \n"
        "vbroadcastsd 8(%[A]), %%ymm0                \n"
        "vfmadd231pd %%ymm1, %%ymm0, %%ymm7          \n"
        "vfmadd231pd %%ymm2, %%ymm0, %%ymm8          \n"
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm9          \n"
        "vbroadcastsd 16(%[A]), %%ymm0               \n"
        "vfmadd231pd %%ymm1, %%ymm0, %%ymm10         \n"
        "vfmadd231pd %%ymm2, %%ymm0, %%ymm11         \n"
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm12         \n"
        "vbroadcastsd 24(%[A]), %%ymm0               \n"
        "vfmadd231pd %%ymm1, %%ymm0, %%ymm13         \n"
        "vfmadd231pd %%ymm2, %%ymm0, %%ymm14         \n"
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm15         \n"
        "addq $32, %[A]                              \n"
        "addq $96, %[B]                              \n"
        "decq %[k]                                   \n"
        "jnz 1b                                      \n"

        /* C[i] += alpha * acc, encoded as vfmadd213pd: dest = src1*dest + src2,
           i.e. acc = alpha*acc + old_C, then store acc back to C. */
        "vbroadcastsd %[alpha], %%ymm0               \n"
        "vfmadd213pd  0(%[C]),  %%ymm0, %%ymm4       \n"
        "vfmadd213pd  32(%[C]), %%ymm0, %%ymm5       \n"
        "vfmadd213pd  64(%[C]), %%ymm0, %%ymm6       \n"
        "vmovupd %%ymm4,  0(%[C])                    \n"
        "vmovupd %%ymm5,  32(%[C])                   \n"
        "vmovupd %%ymm6,  64(%[C])                   \n"
        "addq %[ldc8], %[C]                          \n"
        "vfmadd213pd  0(%[C]),  %%ymm0, %%ymm7       \n"
        "vfmadd213pd  32(%[C]), %%ymm0, %%ymm8       \n"
        "vfmadd213pd  64(%[C]), %%ymm0, %%ymm9       \n"
        "vmovupd %%ymm7,  0(%[C])                    \n"
        "vmovupd %%ymm8,  32(%[C])                   \n"
        "vmovupd %%ymm9,  64(%[C])                   \n"
        "addq %[ldc8], %[C]                          \n"
        "vfmadd213pd  0(%[C]),  %%ymm0, %%ymm10      \n"
        "vfmadd213pd  32(%[C]), %%ymm0, %%ymm11      \n"
        "vfmadd213pd  64(%[C]), %%ymm0, %%ymm12      \n"
        "vmovupd %%ymm10, 0(%[C])                    \n"
        "vmovupd %%ymm11, 32(%[C])                   \n"
        "vmovupd %%ymm12, 64(%[C])                   \n"
        "addq %[ldc8], %[C]                          \n"
        "vfmadd213pd  0(%[C]),  %%ymm0, %%ymm13      \n"
        "vfmadd213pd  32(%[C]), %%ymm0, %%ymm14      \n"
        "vfmadd213pd  64(%[C]), %%ymm0, %%ymm15      \n"
        "vmovupd %%ymm13, 0(%[C])                    \n"
        "vmovupd %%ymm14, 32(%[C])                   \n"
        "vmovupd %%ymm15, 64(%[C])                   \n"
        : [A] "+r" (A_pack), [B] "+r" (B_pack), [C] "+r" (C), [k] "+r" (k_ctr)
        : [alpha] "m" (alpha), [ldc8] "r" ((long)ldc * 8)
        : "ymm0", "ymm1", "ymm2", "ymm3",
          "ymm4", "ymm5", "ymm6", "ymm7",
          "ymm8", "ymm9", "ymm10", "ymm11",
          "ymm12", "ymm13", "ymm14", "ymm15",
          "memory", "cc"
    );
}

static void ukr_4x12_edge(int kc, int mr, int nr,
                          const double* A_pack, const double* B_pack,
                          double* C, int ldc, double alpha) {
    for (int i = 0; i < mr; i++) {
        for (int j = 0; j < nr; j++) {
            double acc = 0.0;
            for (int k = 0; k < kc; k++) {
                acc += A_pack[k * MR_Z + i] * B_pack[k * NR_Z + j];
            }
            C[i * ldc + j] += alpha * acc;
        }
    }
}

void macrokernel_z(int mc, int nc, int kc,
                   const double* A_pack, const double* B_pack,
                   double* C, int ldc) {
    for (int ir = 0; ir < mc; ir += MR_Z) {
        int mr = gemm_imin(MR_Z, mc - ir);
        const double* A_stripe = A_pack + (ir / MR_Z) * MR_Z * kc;
        for (int jr = 0; jr < nc; jr += NR_Z) {
            int nr = gemm_imin(NR_Z, nc - jr);
            const double* B_stripe = B_pack + (jr / NR_Z) * NR_Z * kc;
            double* C_tile = C + ir * ldc + jr;
            if (mr == MR_Z && nr == NR_Z) {
                ukr_4x12_full(kc, A_stripe, B_stripe, C_tile, ldc, 1.0);
            } else {
                ukr_4x12_edge(kc, mr, nr, A_stripe, B_stripe, C_tile, ldc, 1.0);
            }
        }
    }
}

static void gemm_zen3_core(int M, int N, int K, double alpha,
                           const double* A, const double* B, double* C,
                           double* A_pack, double* B_pack) {
    for (int jc = 0; jc < N; jc += NC_Z) {
        int nc = gemm_imin(NC_Z, N - jc);
        for (int pc = 0; pc < K; pc += KC_Z) {
            int kc = gemm_imin(KC_Z, K - pc);
            pack_B_panel_z(kc, nc, B + pc * N + jc, N, B_pack);
            for (int ic = 0; ic < M; ic += MC_Z) {
                int mc = gemm_imin(MC_Z, M - ic);
                pack_A_panel_z(mc, kc, A + ic * K + pc, K, A_pack);

                if (alpha == 1.0) {
                    macrokernel_z(mc, nc, kc, A_pack, B_pack, C + ic * N + jc, N);
                } else {
                    for (int ir = 0; ir < mc; ir += MR_Z) {
                        int mr = gemm_imin(MR_Z, mc - ir);
                        const double* A_stripe = A_pack + (ir / MR_Z) * MR_Z * kc;
                        for (int jr = 0; jr < nc; jr += NR_Z) {
                            int nr = gemm_imin(NR_Z, nc - jr);
                            const double* B_stripe = B_pack + (jr / NR_Z) * NR_Z * kc;
                            double* C_tile = C + (ic + ir) * N + jc + jr;
                            if (mr == MR_Z && nr == NR_Z)
                                ukr_4x12_full(kc, A_stripe, B_stripe, C_tile, N, alpha);
                            else
                                ukr_4x12_edge(kc, mr, nr, A_stripe, B_stripe, C_tile, N, alpha);
                        }
                    }
                }
            }
        }
    }
}

void gemm_zen3(int M, int N, int K, double alpha,
               const double* A, const double* B,
               double beta, double* C) {
    gemm_apply_beta(M, N, beta, C);

    int mb_eff = gemm_round_up(gemm_imin(MC_Z, M), MR_Z);
    int nb_eff = gemm_round_up(gemm_imin(NC_Z, N), NR_Z);
    int kc_eff = gemm_imin(KC_Z, K);

    double* A_pack = (double*)aligned_alloc(64, (size_t)mb_eff * kc_eff * sizeof(double));
    double* B_pack = (double*)aligned_alloc(64, (size_t)kc_eff * nb_eff * sizeof(double));

    gemm_zen3_core(M, N, K, alpha, A, B, C, A_pack, B_pack);

    free(A_pack);
    free(B_pack);
}

void gemm_zen3_omp(int M, int N, int K, double alpha,
                   const double* A, const double* B,
                   double beta, double* C) {
    gemm_apply_beta(M, N, beta, C);

    int mb_eff = gemm_round_up(gemm_imin(MC_Z, M), MR_Z);
    int nb_eff = gemm_round_up(gemm_imin(NC_Z, N), NR_Z);
    int kc_eff = gemm_imin(KC_Z, K);

    #pragma omp parallel
    {
        int nt  = omp_get_num_threads();
        int tid = omp_get_thread_num();

        int chunk = gemm_round_up((M + nt - 1) / nt, MR_Z);
        int m_start = tid * chunk;
        int m_end   = gemm_imin(M, m_start + chunk);

        if (m_start < m_end) {
            double* A_pack = (double*)aligned_alloc(64, (size_t)mb_eff * kc_eff * sizeof(double));
            double* B_pack = (double*)aligned_alloc(64, (size_t)kc_eff * nb_eff * sizeof(double));

            gemm_zen3_core(m_end - m_start, N, K, alpha,
                           A + (size_t)m_start * K, B,
                           C + (size_t)m_start * N,
                           A_pack, B_pack);

            free(A_pack);
            free(B_pack);
        }
    }
}

ALWAYS_INLINE HOT void ukr_4x12_unpacked(int K,
                                         const double* __restrict A, int lda,
                                         const double* __restrict B, int ldb,
                                         double* __restrict C, int ldc,
                                         double alpha) {
    __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd(), c02 = _mm256_setzero_pd();
    __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd(), c12 = _mm256_setzero_pd();
    __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd(), c22 = _mm256_setzero_pd();
    __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd(), c32 = _mm256_setzero_pd();

    const double* a0 = A + 0 * lda;
    const double* a1 = A + 1 * lda;
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
    _mm256_storeu_pd(&C[0*ldc+0], _mm256_fmadd_pd(av, c00, _mm256_loadu_pd(&C[0*ldc+0])));
    _mm256_storeu_pd(&C[0*ldc+4], _mm256_fmadd_pd(av, c01, _mm256_loadu_pd(&C[0*ldc+4])));
    _mm256_storeu_pd(&C[0*ldc+8], _mm256_fmadd_pd(av, c02, _mm256_loadu_pd(&C[0*ldc+8])));
    _mm256_storeu_pd(&C[1*ldc+0], _mm256_fmadd_pd(av, c10, _mm256_loadu_pd(&C[1*ldc+0])));
    _mm256_storeu_pd(&C[1*ldc+4], _mm256_fmadd_pd(av, c11, _mm256_loadu_pd(&C[1*ldc+4])));
    _mm256_storeu_pd(&C[1*ldc+8], _mm256_fmadd_pd(av, c12, _mm256_loadu_pd(&C[1*ldc+8])));
    _mm256_storeu_pd(&C[2*ldc+0], _mm256_fmadd_pd(av, c20, _mm256_loadu_pd(&C[2*ldc+0])));
    _mm256_storeu_pd(&C[2*ldc+4], _mm256_fmadd_pd(av, c21, _mm256_loadu_pd(&C[2*ldc+4])));
    _mm256_storeu_pd(&C[2*ldc+8], _mm256_fmadd_pd(av, c22, _mm256_loadu_pd(&C[2*ldc+8])));
    _mm256_storeu_pd(&C[3*ldc+0], _mm256_fmadd_pd(av, c30, _mm256_loadu_pd(&C[3*ldc+0])));
    _mm256_storeu_pd(&C[3*ldc+4], _mm256_fmadd_pd(av, c31, _mm256_loadu_pd(&C[3*ldc+4])));
    _mm256_storeu_pd(&C[3*ldc+8], _mm256_fmadd_pd(av, c32, _mm256_loadu_pd(&C[3*ldc+8])));
}

static void ukr_scalar_edge_z(int K, int mr, int nr,
                              const double* A, int lda,
                              const double* B, int ldb,
                              double* C, int ldc, double alpha) {
    for (int i = 0; i < mr; i++) {
        for (int j = 0; j < nr; j++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++)
                acc += A[i * lda + k] * B[k * ldb + j];
            C[i * ldc + j] += alpha * acc;
        }
    }
}

HOT void gemm_zen3_micro(int M, int N, int K, double alpha,
                         const double* A, const double* B,
                         double beta, double* C) {
    gemm_apply_beta(M, N, beta, C);

    int mb = (M / MR_Z) * MR_Z;
    int nb = (N / NR_Z) * NR_Z;

    for (int i = 0; i < mb; i += MR_Z) {
        for (int j = 0; j < nb; j += NR_Z) {
            ukr_4x12_unpacked(K, &A[i * K], K, &B[j], N, &C[i * N + j], N, alpha);
        }
        if (nb < N) {
            ukr_scalar_edge_z(K, MR_Z, N - nb,
                              &A[i * K], K, &B[nb], N,
                              &C[i * N + nb], N, alpha);
        }
    }
    if (mb < M) {
        ukr_scalar_edge_z(K, M - mb, N,
                          &A[mb * K], K, B, N,
                          &C[mb * N], N, alpha);
    }
}
