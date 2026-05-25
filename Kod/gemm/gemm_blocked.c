#include "gemm.h"

#define BM 64
#define BN 64
#define BK 256

static inline int imin(int a, int b) { return a < b ? a : b; }

void gemm_blocked(int M, int N, int K, double alpha,
                  const double* A, const double* B,
                  double beta, double* C) {
    if (beta == 0.0) {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                C[i * N + j] = 0.0;
    } else if (beta != 1.0) {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                C[i * N + j] *= beta;
    }

    for (int jc = 0; jc < N; jc += BN) {
        int nc = imin(BN, N - jc);
        for (int pc = 0; pc < K; pc += BK) {
            int kc = imin(BK, K - pc);
            for (int ic = 0; ic < M; ic += BM) {
                int mc = imin(BM, M - ic);
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
