#include "gemm.h"
#include "gemm_internal.h"
#include <immintrin.h>
#include <stdlib.h>

#define ALWAYS_INLINE static inline __attribute__((always_inline))
#define HOT __attribute__((hot))

/* Mikrojadro 4x12 zapisane w funkcjach wbudowanych (intrinsics) -- blizniak
 * asemblerowego ukr_4x12_full z gemm_zen3.c. Rozni sie WYLACZNIE sposobem
 * zapisania petli po K: ten sam packing, te same bloki (MC/NC/KC), ta sama
 * obsluga brzegow. Mikrojadro utrzymuje 12 akumulatorow + 3 wektory B + 1
 * rejestr powielenia A == 16 zywych rejestrow YMM (caly plik architektoniczny).
 * Kazda dodatkowa wartosc tymczasowa, ktora wprowadzi kompilator (np. stala
 * zerowa), podnosi liczbe zywych wartosci powyzej 16 i wymusza spilling
 * akumulatorow na stos -- porownanie z wersja asemblerowa omowiono w rozdziale
 * o wynikach. */
ALWAYS_INLINE HOT void ukr_4x12_intrin(int kc,
                                       const double* __restrict A_pack,
                                       const double* __restrict B_pack,
                                       double* __restrict C, int ldc,
                                       double alpha) {
    __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd(), c02 = _mm256_setzero_pd();
    __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd(), c12 = _mm256_setzero_pd();
    __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd(), c22 = _mm256_setzero_pd();
    __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd(), c32 = _mm256_setzero_pd();

    /* Panele B sa pakowane do bufora wyrownanego do 64 B; kazdy mikropanel
     * lezy pod adresem wielokrotnosci 32 B, wiec odczyt wyrownany jest poprawny
     * (odpowiednik vmovapd w wersji asemblerowej). */
    for (int k = 0; k < kc; k++) {
        __m256d b0 = _mm256_load_pd(&B_pack[k * NR_Z + 0]);
        __m256d b1 = _mm256_load_pd(&B_pack[k * NR_Z + 4]);
        __m256d b2 = _mm256_load_pd(&B_pack[k * NR_Z + 8]);

        __m256d va = _mm256_broadcast_sd(&A_pack[k * MR_Z + 0]);
        c00 = _mm256_fmadd_pd(va, b0, c00);
        c01 = _mm256_fmadd_pd(va, b1, c01);
        c02 = _mm256_fmadd_pd(va, b2, c02);

        va = _mm256_broadcast_sd(&A_pack[k * MR_Z + 1]);
        c10 = _mm256_fmadd_pd(va, b0, c10);
        c11 = _mm256_fmadd_pd(va, b1, c11);
        c12 = _mm256_fmadd_pd(va, b2, c12);

        va = _mm256_broadcast_sd(&A_pack[k * MR_Z + 2]);
        c20 = _mm256_fmadd_pd(va, b0, c20);
        c21 = _mm256_fmadd_pd(va, b1, c21);
        c22 = _mm256_fmadd_pd(va, b2, c22);

        va = _mm256_broadcast_sd(&A_pack[k * MR_Z + 3]);
        c30 = _mm256_fmadd_pd(va, b0, c30);
        c31 = _mm256_fmadd_pd(va, b1, c31);
        c32 = _mm256_fmadd_pd(va, b2, c32);
    }

    /* Epilog: C = alpha * acc + C (zapisy niewyrownane, jak vmovupd w asm). */
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

static void ukr_4x12_intrin_edge(int kc, int mr, int nr,
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

static void macrokernel_intrin(int mc, int nc, int kc,
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
                ukr_4x12_intrin(kc, A_stripe, B_stripe, C_tile, ldc, 1.0);
            else
                ukr_4x12_intrin_edge(kc, mr, nr, A_stripe, B_stripe, C_tile, ldc, 1.0);
        }
    }
}

static void gemm_zen3_intrin_core(int M, int N, int K, double alpha,
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
                    macrokernel_intrin(mc, nc, kc, A_pack, B_pack, C + ic * N + jc, N);
                } else {
                    for (int ir = 0; ir < mc; ir += MR_Z) {
                        int mr = gemm_imin(MR_Z, mc - ir);
                        const double* A_stripe = A_pack + (ir / MR_Z) * MR_Z * kc;
                        for (int jr = 0; jr < nc; jr += NR_Z) {
                            int nr = gemm_imin(NR_Z, nc - jr);
                            const double* B_stripe = B_pack + (jr / NR_Z) * NR_Z * kc;
                            double* C_tile = C + (ic + ir) * N + jc + jr;
                            if (mr == MR_Z && nr == NR_Z)
                                ukr_4x12_intrin(kc, A_stripe, B_stripe, C_tile, N, alpha);
                            else
                                ukr_4x12_intrin_edge(kc, mr, nr, A_stripe, B_stripe, C_tile, N, alpha);
                        }
                    }
                }
            }
        }
    }
}

/* Wariant jednowatkowy mikrojadra 4x12 z funkcji wbudowanych (intrinsics).
 * Identyczny z gemm_zen3 (ST 4x12 packed) poza tym, ze petla K jest w C, nie
 * w asemblerze -- sluzy do zmierzenia kosztu spillingu na stos. */
void gemm_zen3_intrin(int M, int N, int K, double alpha,
                      const double* A, const double* B,
                      double beta, double* C) {
    gemm_apply_beta(M, N, beta, C);

    int mb_eff = gemm_round_up(gemm_imin(MC_Z, M), MR_Z);
    int nb_eff = gemm_round_up(gemm_imin(NC_Z, N), NR_Z);
    int kc_eff = gemm_imin(KC_Z, K);

    double* A_pack = (double*)aligned_alloc(64, (size_t)mb_eff * kc_eff * sizeof(double));
    double* B_pack = (double*)aligned_alloc(64, (size_t)kc_eff * nb_eff * sizeof(double));

    gemm_zen3_intrin_core(M, N, K, alpha, A, B, C, A_pack, B_pack);

    free(A_pack);
    free(B_pack);
}
