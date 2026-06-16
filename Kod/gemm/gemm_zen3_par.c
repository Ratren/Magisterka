#include "gemm.h"
#include "gemm_internal.h"
#include <omp.h>
#include <stdlib.h>

void gemm_zen3_par_omp(int M, int N, int K, double alpha,
                       const double* A, const double* B,
                       double beta, double* C) {
    if (alpha != 1.0) {
        gemm_zen3_omp(M, N, K, alpha, A, B, beta, C);
        return;
    }
    gemm_apply_beta(M, N, beta, C);

    int nthreads = omp_get_max_threads();
    int nb_eff = gemm_round_up(gemm_imin(NC_Z, N), NR_Z);
    int kc_eff = gemm_imin(KC_Z, K);
    int mb_eff = gemm_round_up(gemm_imin(MC_Z, M), MR_Z);

    double* B_pack = (double*)aligned_alloc(64, (size_t)kc_eff * nb_eff * sizeof(double));
    double** A_packs = (double**)malloc((size_t)nthreads * sizeof(double*));
    for (int t = 0; t < nthreads; t++)
        A_packs[t] = (double*)aligned_alloc(64, (size_t)mb_eff * kc_eff * sizeof(double));

    for (int jc = 0; jc < N; jc += NC_Z) {
        int nc = gemm_imin(NC_Z, N - jc);
        for (int pc = 0; pc < K; pc += KC_Z) {
            int kc = gemm_imin(KC_Z, K - pc);

            pack_B_panel_z(kc, nc, B + pc * N + jc, N, B_pack);

            int num_ic = (M + MC_Z - 1) / MC_Z;
            #pragma omp parallel for schedule(static)
            for (int b = 0; b < num_ic; b++) {
                int ic = b * MC_Z;
                int mc = gemm_imin(MC_Z, M - ic);
                double* A_pack = A_packs[omp_get_thread_num()];

                pack_A_panel_z(mc, kc, A + ic * K + pc, K, A_pack);
                macrokernel_z(mc, nc, kc, A_pack, B_pack, C + ic * N + jc, N);
            }
        }
    }

    free(B_pack);
    for (int t = 0; t < nthreads; t++) free(A_packs[t]);
    free(A_packs);
}
