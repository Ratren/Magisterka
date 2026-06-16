#include "gemm.h"
#include "gemm_internal.h"

#define BM 64
#define BN 64
#define BK 256

void gemm_blocked(int M, int N, int K, double alpha,
                  const double* A, const double* B,
                  double beta, double* C) {
    gemm_apply_beta(M, N, beta, C);

    for (int jc = 0; jc < N; jc += BN) {
        int nc = gemm_imin(BN, N - jc);
        for (int pc = 0; pc < K; pc += BK) {
            int kc = gemm_imin(BK, K - pc);
            for (int ic = 0; ic < M; ic += BM) {
                int mc = gemm_imin(BM, M - ic);
                for (int i = 0; i < mc; i++) {
                    for (int k = 0; k < kc; k++) {
                        double a_ik = alpha * A[(ic + i) * K + (pc + k)];
                        const double* b_row = &B[(pc + k) * N + jc];
                        double* c_row = &C[(ic + i) * N + jc];
                        for (int j = 0; j < nc; j++) {
                            c_row[j] += a_ik * b_row[j];
                        }
                    }
                }
            }
        }
    }
}
