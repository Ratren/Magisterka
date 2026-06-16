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

void gemm_zen3_micro  (int M, int N, int K, double alpha,
                       const double* A, const double* B,
                       double beta, double* C);

void gemm_zen3        (int M, int N, int K, double alpha,
                       const double* A, const double* B,
                       double beta, double* C);

void gemm_zen3_intrin (int M, int N, int K, double alpha,
                       const double* A, const double* B,
                       double beta, double* C);

void gemm_zen3_omp    (int M, int N, int K, double alpha,
                       const double* A, const double* B,
                       double beta, double* C);

void gemm_zen3_sched  (int M, int N, int K, double alpha,
                       const double* A, const double* B,
                       double beta, double* C);

void gemm_zen3_sched_omp(int M, int N, int K, double alpha,
                       const double* A, const double* B,
                       double beta, double* C);

void gemm_zen3_par_omp(int M, int N, int K, double alpha,
                       const double* A, const double* B,
                       double beta, double* C);

void gemm_zen3_tuned  (int M, int N, int K, double alpha,
                       const double* A, const double* B,
                       double beta, double* C);

void gemm_zen3_tuned_omp(int M, int N, int K, double alpha,
                       const double* A, const double* B,
                       double beta, double* C);

void gemm_zen3_tiny   (int M, int N, int K, double alpha,
                       const double* A, const double* B,
                       double beta, double* C);

void gemm_zen3_best_omp(int M, int N, int K, double alpha,
                       const double* A, const double* B,
                       double beta, double* C);

void gemm_strassen    (int M, int N, int K, double alpha,
                       const double* A, const double* B,
                       double beta, double* C);

void gemm_strassen_omp(int M, int N, int K, double alpha,
                       const double* A, const double* B,
                       double beta, double* C);

#endif
