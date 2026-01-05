#ifndef GEMV_H
#define GEMV_H

// y = alpha * A * x + beta * y
// Macierz A w formacie row-major

void gemv_naive(int rows, int cols, double alpha,
                const double* A, const double* x,
                double beta, double* y);

void gemv_simd(int rows, int cols, double alpha,
               const double* A, const double* x,
               double beta, double* y);

void gemv_simd_prefetch(int rows, int cols, double alpha,
                        const double* A, const double* x,
                        double beta, double* y);

void gemv_avx_fma_blocked(int rows, int cols, double alpha,
                          const double* A, const double* x,
                          double beta, double* y);

void gemv_avx_fma_v2(int rows, int cols, double alpha,
                     const double* A, const double* x,
                     double beta, double* y);

void gemv_avx_fma_v3(int rows, int cols, double alpha,
                     const double* A, const double* x,
                     double beta, double* y);

#endif
