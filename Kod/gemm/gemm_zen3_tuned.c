#include "gemm.h"
#include "gemm_internal.h"
#include "gemm_zen3_pack.h"
#include <omp.h>
#include <stdlib.h>

#define MAX_TUNED_THREADS 64
static double* g_A_packs[MAX_TUNED_THREADS] = {0};
static size_t  g_A_pack_size = 0;
static double* g_B_pack = NULL;
static size_t  g_B_pack_size = 0;
static int     g_n_alloced = 0;

static void ensure_workspace(int nthreads, size_t a_sz, size_t b_sz) {
    if (b_sz > g_B_pack_size) {
        free(g_B_pack);
        g_B_pack = (double*)aligned_alloc(64, b_sz);
        g_B_pack_size = b_sz;
    }
    if (a_sz > g_A_pack_size || nthreads > g_n_alloced) {
        for (int t = 0; t < g_n_alloced; t++) free(g_A_packs[t]);
        for (int t = 0; t < nthreads && t < MAX_TUNED_THREADS; t++)
            g_A_packs[t] = (double*)aligned_alloc(64, a_sz);
        g_A_pack_size = a_sz;
        g_n_alloced = nthreads;
    }
}

void gemm_zen3_tuned(int M, int N, int K, double alpha,
                     const double* A, const double* B,
                     double beta, double* C) {
    int md = M < N ? M : N;
    if (K < md) md = K;
    if (md <= 96) {
        gemm_zen3_tiny(M, N, K, alpha, A, B, beta, C);
        return;
    }
    gemm_zen3(M, N, K, alpha, A, B, beta, C);
}

void gemm_zen3_tuned_omp(int M, int N, int K, double alpha,
                         const double* A, const double* B,
                         double beta, double* C) {
    if (alpha != 1.0) {
        gemm_zen3_omp(M, N, K, alpha, A, B, beta, C);
        return;
    }
    gemm_apply_beta(M, N, beta, C);

    int nthreads = omp_get_max_threads();
    if (nthreads > MAX_TUNED_THREADS) nthreads = MAX_TUNED_THREADS;
    int nb_eff = gemm_round_up(gemm_imin(NC_Z, N), NR_Z);
    int kc_eff = gemm_imin(KC_Z, K);
    int mb_eff = gemm_round_up(gemm_imin(MC_Z, M), MR_Z);

    ensure_workspace(nthreads,
                     (size_t)mb_eff * kc_eff * sizeof(double),
                     (size_t)kc_eff * nb_eff * sizeof(double));

    for (int jc = 0; jc < N; jc += NC_Z) {
        int nc = gemm_imin(NC_Z, N - jc);
        for (int pc = 0; pc < K; pc += KC_Z) {
            int kc = gemm_imin(KC_Z, K - pc);

            pack_B_panel_v(kc, nc, B + pc * N + jc, N, g_B_pack);

            int num_ic = (M + MC_Z - 1) / MC_Z;
            #pragma omp parallel for schedule(static)
            for (int b = 0; b < num_ic; b++) {
                int ic = b * MC_Z;
                int mc = gemm_imin(MC_Z, M - ic);
                int tid = omp_get_thread_num();
                double* A_pack = g_A_packs[tid];

                pack_A_panel_v(mc, kc, A + ic * K + pc, K, A_pack);
                macrokernel_z(mc, nc, kc, A_pack, g_B_pack, C + ic * N + jc, N);
            }
        }
    }
}
