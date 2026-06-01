#include "gemm.h"
#include "gemm_internal.h"
#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>

#define ALWAYS_INLINE static inline __attribute__((always_inline))
#define HOT __attribute__((hot))

ALWAYS_INLINE HOT void ukr_4x12_sched(int kc,
                                      const double* __restrict A_pack,
                                      const double* __restrict B_pack,
                                      double* __restrict C, int ldc,
                                      double alpha) {
    long k_iter = kc / 4;
    long k_left = kc - k_iter * 4;
    long ldc8   = (long)ldc * 8;

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

        /* prologue: pre-load B for first K iteration */
        "vmovapd 0(%[B]),  %%ymm1                    \n"
        "vmovapd 32(%[B]), %%ymm2                    \n"
        "vmovapd 64(%[B]), %%ymm3                    \n"

        "testq %[kiter], %[kiter]                    \n"
        "jz 2f                                       \n"

        /* main loop body: 4 K iterations per pass */
        "1:                                          \n"
        /* --- K iter 0 --- (consumes pre-loaded ymm1,ymm2,ymm3) */
        "vbroadcastsd 0(%[A]),  %%ymm0               \n"
        "vfmadd231pd %%ymm1, %%ymm0, %%ymm4          \n"
        "vfmadd231pd %%ymm2, %%ymm0, %%ymm5          \n"
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm6          \n"
        "vbroadcastsd 8(%[A]),  %%ymm0               \n"
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
        /* pre-load B for K iter 1 (issued while iter-0 FMAs still in flight) */
        "vmovapd 96(%[B]),  %%ymm1                   \n"
        "vmovapd 128(%[B]), %%ymm2                   \n"
        "vmovapd 160(%[B]), %%ymm3                   \n"

        /* --- K iter 1 --- */
        "vbroadcastsd 32(%[A]), %%ymm0               \n"
        "vfmadd231pd %%ymm1, %%ymm0, %%ymm4          \n"
        "vfmadd231pd %%ymm2, %%ymm0, %%ymm5          \n"
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm6          \n"
        "vbroadcastsd 40(%[A]), %%ymm0               \n"
        "vfmadd231pd %%ymm1, %%ymm0, %%ymm7          \n"
        "vfmadd231pd %%ymm2, %%ymm0, %%ymm8          \n"
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm9          \n"
        "vbroadcastsd 48(%[A]), %%ymm0               \n"
        "vfmadd231pd %%ymm1, %%ymm0, %%ymm10         \n"
        "vfmadd231pd %%ymm2, %%ymm0, %%ymm11         \n"
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm12         \n"
        "vbroadcastsd 56(%[A]), %%ymm0               \n"
        "vfmadd231pd %%ymm1, %%ymm0, %%ymm13         \n"
        "vfmadd231pd %%ymm2, %%ymm0, %%ymm14         \n"
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm15         \n"
        "vmovapd 192(%[B]), %%ymm1                   \n"
        "vmovapd 224(%[B]), %%ymm2                   \n"
        "vmovapd 256(%[B]), %%ymm3                   \n"

        /* --- K iter 2 --- */
        "vbroadcastsd 64(%[A]), %%ymm0               \n"
        "vfmadd231pd %%ymm1, %%ymm0, %%ymm4          \n"
        "vfmadd231pd %%ymm2, %%ymm0, %%ymm5          \n"
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm6          \n"
        "vbroadcastsd 72(%[A]), %%ymm0               \n"
        "vfmadd231pd %%ymm1, %%ymm0, %%ymm7          \n"
        "vfmadd231pd %%ymm2, %%ymm0, %%ymm8          \n"
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm9          \n"
        "vbroadcastsd 80(%[A]), %%ymm0               \n"
        "vfmadd231pd %%ymm1, %%ymm0, %%ymm10         \n"
        "vfmadd231pd %%ymm2, %%ymm0, %%ymm11         \n"
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm12         \n"
        "vbroadcastsd 88(%[A]), %%ymm0               \n"
        "vfmadd231pd %%ymm1, %%ymm0, %%ymm13         \n"
        "vfmadd231pd %%ymm2, %%ymm0, %%ymm14         \n"
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm15         \n"
        "vmovapd 288(%[B]), %%ymm1                   \n"
        "vmovapd 320(%[B]), %%ymm2                   \n"
        "vmovapd 352(%[B]), %%ymm3                   \n"

        /* --- K iter 3 --- */
        "vbroadcastsd 96(%[A]),  %%ymm0              \n"
        "vfmadd231pd %%ymm1, %%ymm0, %%ymm4          \n"
        "vfmadd231pd %%ymm2, %%ymm0, %%ymm5          \n"
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm6          \n"
        "vbroadcastsd 104(%[A]), %%ymm0              \n"
        "vfmadd231pd %%ymm1, %%ymm0, %%ymm7          \n"
        "vfmadd231pd %%ymm2, %%ymm0, %%ymm8          \n"
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm9          \n"
        "vbroadcastsd 112(%[A]), %%ymm0              \n"
        "vfmadd231pd %%ymm1, %%ymm0, %%ymm10         \n"
        "vfmadd231pd %%ymm2, %%ymm0, %%ymm11         \n"
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm12         \n"
        "vbroadcastsd 120(%[A]), %%ymm0              \n"
        "vfmadd231pd %%ymm1, %%ymm0, %%ymm13         \n"
        "vfmadd231pd %%ymm2, %%ymm0, %%ymm14         \n"
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm15         \n"
        /* advance pointers; pre-load B for K iter 0 of the next pass */
        "addq $128, %[A]                             \n"
        "addq $384, %[B]                             \n"
        "vmovapd 0(%[B]),  %%ymm1                    \n"
        "vmovapd 32(%[B]), %%ymm2                    \n"
        "vmovapd 64(%[B]), %%ymm3                    \n"
        "decq %[kiter]                               \n"
        "jnz 1b                                      \n"

        /* k_left edge loop: 0..3 unrolled K iterations */
        "2:                                          \n"
        "testq %[kleft], %[kleft]                    \n"
        "jz 3f                                       \n"
        "4:                                          \n"
        "vbroadcastsd 0(%[A]),  %%ymm0               \n"
        "vfmadd231pd %%ymm1, %%ymm0, %%ymm4          \n"
        "vfmadd231pd %%ymm2, %%ymm0, %%ymm5          \n"
        "vfmadd231pd %%ymm3, %%ymm0, %%ymm6          \n"
        "vbroadcastsd 8(%[A]),  %%ymm0               \n"
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
        "vmovapd 0(%[B]),  %%ymm1                    \n"
        "vmovapd 32(%[B]), %%ymm2                    \n"
        "vmovapd 64(%[B]), %%ymm3                    \n"
        "decq %[kleft]                               \n"
        "jnz 4b                                      \n"

        /* epilog: C += alpha * acc using vfmadd213pd (dest = src1*dest + src2) */
        "3:                                          \n"
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
        : [A] "+r" (A_pack), [B] "+r" (B_pack), [C] "+r" (C),
          [kiter] "+r" (k_iter), [kleft] "+r" (k_left)
        : [alpha] "m" (alpha), [ldc8] "r" (ldc8)
        : "ymm0", "ymm1", "ymm2", "ymm3",
          "ymm4", "ymm5", "ymm6", "ymm7",
          "ymm8", "ymm9", "ymm10", "ymm11",
          "ymm12", "ymm13", "ymm14", "ymm15",
          "memory", "cc"
    );
}

static void ukr_4x12_sched_edge(int kc, int mr, int nr,
                                const double* A_pack, const double* B_pack,
                                double* C, int ldc, double alpha) {
    for (int i = 0; i < mr; i++) {
        for (int j = 0; j < nr; j++) {
            double acc = 0.0;
            for (int k = 0; k < kc; k++)
                acc += A_pack[k * MR_Z + i] * B_pack[k * NR_Z + j];
            C[i * ldc + j] += alpha * acc;
        }
    }
}

static void macrokernel_sched(int mc, int nc, int kc,
                              const double* A_pack, const double* B_pack,
                              double* C, int ldc) {
    for (int ir = 0; ir < mc; ir += MR_Z) {
        int mr = gemm_imin(MR_Z, mc - ir);
        const double* A_stripe = A_pack + (ir / MR_Z) * MR_Z * kc;
        for (int jr = 0; jr < nc; jr += NR_Z) {
            int nr = gemm_imin(NR_Z, nc - jr);
            const double* B_stripe = B_pack + (jr / NR_Z) * NR_Z * kc;
            double* C_tile = C + ir * ldc + jr;
            if (mr == MR_Z && nr == NR_Z)
                ukr_4x12_sched(kc, A_stripe, B_stripe, C_tile, ldc, 1.0);
            else
                ukr_4x12_sched_edge(kc, mr, nr, A_stripe, B_stripe, C_tile, ldc, 1.0);
        }
    }
}

static void gemm_zen3_sched_core(int M, int N, int K, double alpha,
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
                    macrokernel_sched(mc, nc, kc, A_pack, B_pack, C + ic * N + jc, N);
                } else {
                    for (int ir = 0; ir < mc; ir += MR_Z) {
                        int mr = gemm_imin(MR_Z, mc - ir);
                        const double* A_stripe = A_pack + (ir / MR_Z) * MR_Z * kc;
                        for (int jr = 0; jr < nc; jr += NR_Z) {
                            int nr = gemm_imin(NR_Z, nc - jr);
                            const double* B_stripe = B_pack + (jr / NR_Z) * NR_Z * kc;
                            double* C_tile = C + (ic + ir) * N + jc + jr;
                            if (mr == MR_Z && nr == NR_Z)
                                ukr_4x12_sched(kc, A_stripe, B_stripe, C_tile, N, alpha);
                            else
                                ukr_4x12_sched_edge(kc, mr, nr, A_stripe, B_stripe, C_tile, N, alpha);
                        }
                    }
                }
            }
        }
    }
}

void gemm_zen3_sched(int M, int N, int K, double alpha,
                     const double* A, const double* B,
                     double beta, double* C) {
    gemm_apply_beta(M, N, beta, C);

    int mb_eff = gemm_round_up(gemm_imin(MC_Z, M), MR_Z);
    int nb_eff = gemm_round_up(gemm_imin(NC_Z, N), NR_Z);
    int kc_eff = gemm_imin(KC_Z, K);

    double* A_pack = (double*)aligned_alloc(64, (size_t)mb_eff * kc_eff * sizeof(double));
    double* B_pack = (double*)aligned_alloc(64, (size_t)kc_eff * nb_eff * sizeof(double));

    gemm_zen3_sched_core(M, N, K, alpha, A, B, C, A_pack, B_pack);

    free(A_pack);
    free(B_pack);
}

void gemm_zen3_sched_omp(int M, int N, int K, double alpha,
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

            gemm_zen3_sched_core(m_end - m_start, N, K, alpha,
                                 A + (size_t)m_start * K, B,
                                 C + (size_t)m_start * N,
                                 A_pack, B_pack);

            free(A_pack);
            free(B_pack);
        }
    }
}
