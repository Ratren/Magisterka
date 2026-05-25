#ifndef CONV_INTERNAL_H
#define CONV_INTERNAL_H

// Direct conv microkernel tile: 6 output spatial positions x 16 output channels.
// 6 broadcasts of X + 2 ymm of Wk + 12 FMAs per (ic, kh, kw) step.
#define CONV_MR 6
#define CONV_NR 16

// sgemm microkernel (used by conv_im2col_gemm): 6 x 16 float tile.
#define SMR 6
#define SNR 16
#define SMC 96
#define SKC 256
#define SNC 4096

void im2col(int Cin, int H, int W, int KH, int KW,
            const float* X, float* col);

void sgemm_packed_internal(int M, int N, int K, float alpha,
                           const float* A, const float* B,
                           float beta, float* C);

#endif
