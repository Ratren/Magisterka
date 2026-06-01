#ifndef GEMM_ZEN3_PACK_H
#define GEMM_ZEN3_PACK_H

#include "gemm_internal.h"
#include <immintrin.h>

static inline __attribute__((always_inline))
void pack_A_4rows_v(int kc, const double* A, int lda, double* dst) {
    const double* r0 = A;
    const double* r1 = A + lda;
    const double* r2 = A + 2 * lda;
    const double* r3 = A + 3 * lda;
    int k = 0;
    for (; k + 4 <= kc; k += 4) {
        __m256d a0 = _mm256_loadu_pd(&r0[k]);
        __m256d a1 = _mm256_loadu_pd(&r1[k]);
        __m256d a2 = _mm256_loadu_pd(&r2[k]);
        __m256d a3 = _mm256_loadu_pd(&r3[k]);
        __m256d t0 = _mm256_unpacklo_pd(a0, a1);
        __m256d t1 = _mm256_unpackhi_pd(a0, a1);
        __m256d t2 = _mm256_unpacklo_pd(a2, a3);
        __m256d t3 = _mm256_unpackhi_pd(a2, a3);
        _mm256_store_pd(&dst[(k+0)*MR_Z], _mm256_permute2f128_pd(t0, t2, 0x20));
        _mm256_store_pd(&dst[(k+1)*MR_Z], _mm256_permute2f128_pd(t1, t3, 0x20));
        _mm256_store_pd(&dst[(k+2)*MR_Z], _mm256_permute2f128_pd(t0, t2, 0x31));
        _mm256_store_pd(&dst[(k+3)*MR_Z], _mm256_permute2f128_pd(t1, t3, 0x31));
    }
    for (; k < kc; k++) {
        dst[k*MR_Z+0] = r0[k]; dst[k*MR_Z+1] = r1[k];
        dst[k*MR_Z+2] = r2[k]; dst[k*MR_Z+3] = r3[k];
    }
}

/* Pack one NR_Z=12 wide B stripe: 3 contiguous 256-bit copies per K row. */
static inline __attribute__((always_inline))
void pack_B_12cols_v(int kc, const double* B, int ldb, double* dst) {
    for (int k = 0; k < kc; k++) {
        _mm256_store_pd(&dst[k*NR_Z + 0], _mm256_loadu_pd(&B[k * ldb + 0]));
        _mm256_store_pd(&dst[k*NR_Z + 4], _mm256_loadu_pd(&B[k * ldb + 4]));
        _mm256_store_pd(&dst[k*NR_Z + 8], _mm256_loadu_pd(&B[k * ldb + 8]));
    }
}

/* Full A-panel pack with edge handling for the trailing mr<MR_Z stripe. */
static inline void pack_A_panel_v(int mc, int kc, const double* A, int lda, double* A_pack) {
    int stripes = mc / MR_Z;
    for (int s = 0; s < stripes; s++)
        pack_A_4rows_v(kc, A + s * MR_Z * lda, lda, A_pack + s * MR_Z * kc);
    if (stripes * MR_Z < mc) {
        int i_base = stripes * MR_Z;
        double* dst = A_pack + stripes * MR_Z * kc;
        for (int k = 0; k < kc; k++)
            for (int i = 0; i < MR_Z; i++)
                dst[k * MR_Z + i] = (i_base + i < mc) ? A[(i_base + i) * lda + k] : 0.0;
    }
}

/* Full B-panel pack with edge handling for the trailing nr<NR_Z stripe. */
static inline void pack_B_panel_v(int kc, int nc, const double* B, int ldb, double* B_pack) {
    int stripes = nc / NR_Z;
    for (int s = 0; s < stripes; s++)
        pack_B_12cols_v(kc, B + s * NR_Z, ldb, B_pack + s * NR_Z * kc);
    if (stripes * NR_Z < nc) {
        int j_base = stripes * NR_Z;
        double* dst = B_pack + stripes * NR_Z * kc;
        for (int k = 0; k < kc; k++)
            for (int j = 0; j < NR_Z; j++)
                dst[k * NR_Z + j] = (j_base + j < nc) ? B[k * ldb + j_base + j] : 0.0;
    }
}

#endif
