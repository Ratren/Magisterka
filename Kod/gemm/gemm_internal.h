#ifndef GEMM_INTERNAL_H
#define GEMM_INTERNAL_H

#define MR 6
#define NR 8
#define MC 72
#define KC 256
#define NC 4080

void dgemm_ukr_6x8(int kc,
                   const double* __restrict A_pack,
                   const double* __restrict B_pack,
                   double* __restrict C, int ldc);

void dgemm_ukr_6x8_edge(int kc, int mr, int nr,
                        const double* __restrict A_pack,
                        const double* __restrict B_pack,
                        double* __restrict C, int ldc);

void pack_A_panel(int mc, int kc,
                  const double* A, int lda,
                  double* A_pack);

void pack_B_panel(int kc, int nc,
                  const double* B, int ldb,
                  double* B_pack);

void macrokernel(int mc, int nc, int kc,
                 const double* A_pack, const double* B_pack,
                 double* C, int ldc);

#endif
