#include "gemm.h"
#include "gemm_internal.h"
#include "gemm_zen3_pack.h"
#include <omp.h>
#include <stdlib.h>

#define MAX_BEST_THREADS 32

/* Best multi-thread DGEMM for Zen 3 — bundles every win found in this codebase:
   - Tiny dispatch (min(M,N,K) <= 96 -> gemm_zen3_tiny, no alloc/packing)
   - Shared B_pack across threads (one copy, fits L3 even at 4096^3)
   - Per-thread A_pack (small, fits L2)
   - Persistent workspace (no aligned_alloc per call)
   - Vectorised 4x4-transpose A packer + 3-vector B packer (gemm_zen3_pack.h)
   - Parallel B packing when NC is large enough to amortise the barrier
   - ic-loop parallelism (BLIS-style multi-loop) with M-split fallback when
     there aren't enough ic-blocks to keep all threads busy. */

static double* g_A6_packs[MAX_BEST_THREADS] = {0};
static size_t  g_A6_size = 0;
static double* g_B6_pack = NULL;
static size_t  g_B6_size = 0;
static int     g_T6_alloced = 0;

static void ws_ensure_b(int nt, size_t a_sz, size_t b_sz) {
    if (b_sz > g_B6_size) {
        free(g_B6_pack);
        g_B6_pack = (double*)aligned_alloc(64, b_sz);
        g_B6_size = b_sz;
    }
    if (a_sz > g_A6_size || nt > g_T6_alloced) {
        for (int t = 0; t < g_T6_alloced; t++) free(g_A6_packs[t]);
        for (int t = 0; t < nt && t < MAX_BEST_THREADS; t++)
            g_A6_packs[t] = (double*)aligned_alloc(64, a_sz);
        g_A6_size = a_sz;
        g_T6_alloced = nt;
    }
}

/* Pack a single NR_Z-wide B stripe -- either the full vectorised path or the
   zero-padded scalar edge for the final partial stripe. Used by the parallel
   B pack inside gemm_zen3_best_omp. */
static inline void pack_B_one_stripe(int s, int kc, int nc,
                                     const double* B, int ldb, double* B_pack) {
    int j_base = s * NR_Z;
    int width  = gemm_imin(NR_Z, nc - j_base);
    double* dst = B_pack + (size_t)s * NR_Z * kc;
    if (width == NR_Z) {
        pack_B_12cols_v(kc, B + j_base, ldb, dst);
    } else {
        for (int k = 0; k < kc; k++)
            for (int j = 0; j < NR_Z; j++)
                dst[k * NR_Z + j] = (j < width) ? B[k * ldb + j_base + j] : 0.0;
    }
}

void gemm_zen3_best_omp(int M, int N, int K, double alpha,
                        const double* A, const double* B,
                        double beta, double* C) {
    int nthreads = omp_get_max_threads();
    int md = M < N ? M : N;
    if (K < md) md = K;

    /* Tiny: skip OpenMP overhead and use the unpacked tiny path. At
       md<=96 there's not enough work to amortise thread spawn. */
    if (md <= 96) {
        if (nthreads == 1) {
            gemm_zen3_tiny(M, N, K, alpha, A, B, beta, C);
        } else {
            /* Small but parallel: per-thread M-split (gemm_zen3_omp) keeps all
               threads busy. Shared-B would leave most threads idle. */
            gemm_zen3_omp(M, N, K, alpha, A, B, beta, C);
        }
        return;
    }

    /* Shared-B path requires enough ic-blocks to feed every thread. With M
       below nthreads*MC_Z most threads sit idle (e.g. M=256, MC_Z=192,
       6 threads -> only 2 ic blocks). Fall back to per-thread M-split. */
    int num_ic_blocks = (M + MC_Z - 1) / MC_Z;
    if (num_ic_blocks < nthreads) {
        gemm_zen3_omp(M, N, K, alpha, A, B, beta, C);
        return;
    }

    if (alpha != 1.0) {
        gemm_zen3_omp(M, N, K, alpha, A, B, beta, C);
        return;
    }
    gemm_apply_beta(M, N, beta, C);

    if (nthreads > MAX_BEST_THREADS) nthreads = MAX_BEST_THREADS;
    int nb_eff = gemm_round_up(gemm_imin(NC_Z, N), NR_Z);
    int kc_eff = gemm_imin(KC_Z, K);
    int mb_eff = gemm_round_up(gemm_imin(MC_Z, M), MR_Z);

    ws_ensure_b(nthreads,
                (size_t)mb_eff * kc_eff * sizeof(double),
                (size_t)kc_eff * nb_eff * sizeof(double));

    /* Parallelise B packing only when there are enough stripes to amortise
       the OpenMP barrier cost (~30us at 6 threads). At small NC the
       sequential path is faster. */
    const int PAR_B_PACK_MIN_STRIPES = 64;

    for (int jc = 0; jc < N; jc += NC_Z) {
        int nc = gemm_imin(NC_Z, N - jc);
        int B_stripes = (nc + NR_Z - 1) / NR_Z;
        for (int pc = 0; pc < K; pc += KC_Z) {
            int kc = gemm_imin(KC_Z, K - pc);

            const double* B_pc_jc = B + (size_t)pc * N + jc;
            if (B_stripes >= PAR_B_PACK_MIN_STRIPES) {
                #pragma omp parallel for schedule(static)
                for (int s = 0; s < B_stripes; s++)
                    pack_B_one_stripe(s, kc, nc, B_pc_jc, N, g_B6_pack);
            } else {
                for (int s = 0; s < B_stripes; s++)
                    pack_B_one_stripe(s, kc, nc, B_pc_jc, N, g_B6_pack);
            }

            int num_ic = (M + MC_Z - 1) / MC_Z;
            #pragma omp parallel for schedule(static)
            for (int b = 0; b < num_ic; b++) {
                int ic = b * MC_Z;
                int mc = gemm_imin(MC_Z, M - ic);
                double* A_pack = g_A6_packs[omp_get_thread_num()];

                pack_A_panel_v(mc, kc, A + (size_t)ic * K + pc, K, A_pack);
                macrokernel_z(mc, nc, kc, A_pack, g_B6_pack,
                              C + (size_t)ic * N + jc, N);
            }
        }
    }
}
