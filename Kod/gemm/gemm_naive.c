#include "gemm.h"

void gemm_naive(int M, int N, int K, double alpha,
                const double* A, const double* B,
                double beta, double* C) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * acc + beta * C[i * N + j];
        }
    }
}
