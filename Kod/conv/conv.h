#ifndef CONV_H
#define CONV_H

// 2D convolution, single batch, stride=1, no padding, float32.
// Input  X:  shape (C_in, H, W)              row-major.
// Kernel W:  shape (C_out, C_in, KH, KW)     row-major.
// Output Y:  shape (C_out, OH, OW)           row-major, where
//            OH = H - KH + 1, OW = W - KW + 1.

void conv_naive       (int Cin, int H, int W, int KH, int KW, int Cout,
                       const float* X, const float* Wk, float* Y);

void conv_reorder     (int Cin, int H, int W, int KH, int KW, int Cout,
                       const float* X, const float* Wk, float* Y);

void conv_blocked     (int Cin, int H, int W, int KH, int KW, int Cout,
                       const float* X, const float* Wk, float* Y);

void conv_microkernel (int Cin, int H, int W, int KH, int KW, int Cout,
                       const float* X, const float* Wk, float* Y);

void conv_packed      (int Cin, int H, int W, int KH, int KW, int Cout,
                       const float* X, const float* Wk, float* Y);

void conv_im2col_gemm (int Cin, int H, int W, int KH, int KW, int Cout,
                       const float* X, const float* Wk, float* Y);

void conv_im2col_openblas(int Cin, int H, int W, int KH, int KW, int Cout,
                          const float* X, const float* Wk, float* Y);

void conv_im2col_blis (int Cin, int H, int W, int KH, int KW, int Cout,
                       const float* X, const float* Wk, float* Y);

void conv_omp         (int Cin, int H, int W, int KH, int KW, int Cout,
                       const float* X, const float* Wk, float* Y);

void conv_1x1         (int Cin, int H, int W, int KH, int KW, int Cout,
                       const float* X, const float* Wk, float* Y);

void conv_nchwc       (int Cin, int H, int W, int KH, int KW, int Cout,
                       const float* X, const float* Wk, float* Y);

void conv_winograd    (int Cin, int H, int W, int KH, int KW, int Cout,
                       const float* X, const float* Wk, float* Y);

void conv_libxsmm     (int Cin, int H, int W, int KH, int KW, int Cout,
                       const float* X, const float* Wk, float* Y);

void conv_zen3        (int Cin, int H, int W, int KH, int KW, int Cout,
                       const float* X, const float* Wk, float* Y);

void conv_zen3_omp    (int Cin, int H, int W, int KH, int KW, int Cout,
                       const float* X, const float* Wk, float* Y);

#endif
