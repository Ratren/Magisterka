#ifndef CONV_INTERNAL_H
#define CONV_INTERNAL_H

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

void conv_pack_W_oc_innermost(int Cin, int KH, int KW, int Cout,
                              const float* Wk, float* W_pack);

void sgemm_packed_internal(int M, int N, int K, float alpha,
                           const float* A, const float* B,
                           float beta, float* C);

#endif
