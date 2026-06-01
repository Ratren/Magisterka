#include "conv.h"
#include "conv_internal.h"
#include <stdlib.h>
#include <string.h>
#include "openblas/cblas.h"

typedef void (*blis_sgemm_f77_func)(const char*, const char*,
                                    const int*, const int*, const int*,
                                    const float*, const float*, const int*,
                                    const float*, const int*,
                                    const float*, float*, const int*);

extern blis_sgemm_f77_func conv_blis_sgemm_f77;

void im2col(int Cin, int H, int W, int KH, int KW,
            const float* X, float* col) {
    int OH = H - KH + 1;
    int OW = W - KW + 1;
    for (int ic = 0; ic < Cin; ic++) {
        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                int row = (ic * KH + kh) * KW + kw;
                float* dst = &col[row * OH * OW];
                for (int oh = 0; oh < OH; oh++) {
                    const float* src = &X[ic * H * W + (oh + kh) * W + kw];
                    memcpy(&dst[oh * OW], src, OW * sizeof(float));
                }
            }
        }
    }
}

static float* alloc_im2col_buf(int Cin, int KH, int KW, int OH, int OW) {
    size_t bytes = (size_t)Cin * KH * KW * OH * OW * sizeof(float);
    return (float*)aligned_alloc(64, bytes);
}

void conv_im2col_gemm(int Cin, int H, int W, int KH, int KW, int Cout,
                      const float* X, const float* Wk, float* Y) {
    int OH = H - KH + 1;
    int OW = W - KW + 1;
    int K  = Cin * KH * KW;
    int N  = OH * OW;

    float* col = alloc_im2col_buf(Cin, KH, KW, OH, OW);
    im2col(Cin, H, W, KH, KW, X, col);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                Cout, N, K, 1.0f, Wk, K, col, N, 0.0f, Y, N);

    free(col);
}

void conv_im2col_openblas(int Cin, int H, int W, int KH, int KW, int Cout,
                          const float* X, const float* Wk, float* Y) {
    conv_im2col_gemm(Cin, H, W, KH, KW, Cout, X, Wk, Y);
}

void conv_im2col_blis(int Cin, int H, int W, int KH, int KW, int Cout,
                      const float* X, const float* Wk, float* Y) {
    if (!conv_blis_sgemm_f77) return;

    int OH = H - KH + 1;
    int OW = W - KW + 1;
    int K  = Cin * KH * KW;
    int N  = OH * OW;

    float* col = alloc_im2col_buf(Cin, KH, KW, OH, OW);
    im2col(Cin, H, W, KH, KW, X, col);

    char trans = 'N';
    int m = N, n = Cout, k = K, lda = N, ldb = K, ldc = N;
    float alpha = 1.0f, beta = 0.0f;
    conv_blis_sgemm_f77(&trans, &trans, &m, &n, &k, &alpha,
                        col, &lda, Wk, &ldb, &beta, Y, &ldc);

    free(col);
}
