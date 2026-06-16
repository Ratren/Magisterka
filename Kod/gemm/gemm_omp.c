#include "gemm.h"
#include "gemm_internal.h"
#include <omp.h>
#include <stdlib.h>

extern void gemm_packed_core(int M, int N, int K, double alpha,
                             const double* A, const double* B, double* C,
                             double* A_pack, double* B_pack);

void gemm_omp(int M, int N, int K, double alpha,
              const double* A, const double* B,
              double beta, double* C) {
    gemm_apply_beta(M, N, beta, C);

    int mb_eff = gemm_round_up(gemm_imin(MC, M), MR);
    int nb_eff = gemm_round_up(gemm_imin(NC, N), NR);
    int kc_eff = gemm_imin(KC, K);

    #pragma omp parallel
    {
        int nt   = omp_get_num_threads();
        int tid  = omp_get_thread_num();

        int chunk = gemm_round_up((M + nt - 1) / nt, MR);
        int m_start = tid * chunk;
        int m_end   = gemm_imin(M, m_start + chunk);

        if (m_start < m_end) {
            double* A_pack = (double*)aligned_alloc(64, (size_t)mb_eff * kc_eff * sizeof(double));
            double* B_pack = (double*)aligned_alloc(64, (size_t)kc_eff * nb_eff * sizeof(double));

            gemm_packed_core(m_end - m_start, N, K, alpha,
                             A + (size_t)m_start * K, B,
                             C + (size_t)m_start * N,
                             A_pack, B_pack);

            free(A_pack);
            free(B_pack);
        }
    }
}
