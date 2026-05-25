#include "gemm.h"

void gemm_ikj(int M, int N, int K, double alpha,
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

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            double a_ik = alpha * A[i * K + k];
            const double* b_row = &B[k * N];
            double* c_row = &C[i * N];
            for (int j = 0; j < N; j++) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
}
