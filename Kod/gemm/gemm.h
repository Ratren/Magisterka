#ifndef GEMM_H
#define GEMM_H

void gemm_naive       (int M, int N, int K, double alpha,
                       const double* A, const double* B,
                       double beta, double* C);

void gemm_ikj         (int M, int N, int K, double alpha,
                       const double* A, const double* B,
                       double beta, double* C);

void gemm_blocked     (int M, int N, int K, double alpha,
                       const double* A, const double* B,
                       double beta, double* C);

void gemm_microkernel (int M, int N, int K, double alpha,
                       const double* A, const double* B,
                       double beta, double* C);

void gemm_packed      (int M, int N, int K, double alpha,
                       const double* A, const double* B,
                       double beta, double* C);

void gemm_omp         (int M, int N, int K, double alpha,
                       const double* A, const double* B,
                       double beta, double* C);

#endif
