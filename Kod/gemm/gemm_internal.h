#ifndef GEMM_INTERNAL_H
#define GEMM_INTERNAL_H

#define MR 6
#define NR 8
#define MC 72
#define KC 256
#define NC 4080

static inline int gemm_imin(int a, int b) { return a < b ? a : b; }
static inline int gemm_round_up(int x, int m) { return ((x + m - 1) / m) * m; }

void gemm_apply_beta(int M, int N, double beta, double* C);

void pack_A_panel(int mc, int kc,
                  const double* A, int lda,
                  double* A_pack);

void pack_B_panel(int kc, int nc,
                  const double* B, int ldb,
                  double* B_pack);

void macrokernel(int mc, int nc, int kc,
                 const double* A_pack, const double* B_pack,
                 double* C, int ldc);

#define MR_Z 4
#define NR_Z 12
#define MC_Z 192
#define KC_Z 240
#define NC_Z 4080

void pack_A_panel_z(int mc, int kc,
                    const double* A, int lda,
                    double* A_pack);

void pack_B_panel_z(int kc, int nc,
                    const double* B, int ldb,
                    double* B_pack);

void macrokernel_z(int mc, int nc, int kc,
                   const double* A_pack, const double* B_pack,
                   double* C, int ldc);

#endif
