#include "conv.h"
#include "common.h"
#include <stdlib.h>
#include <string.h>

extern void im2col(int Cin, int H, int W, int KH, int KW,
                   const float* X, float* col);

typedef void (*libxsmm_sgemm_f)(const char*, const char*,
                                const int*, const int*, const int*,
                                const float*, const float*, const int*,
                                const float*, const int*,
                                const float*, float*, const int*);

void conv_libxsmm(int Cin, int H, int W, int KH, int KW, int Cout,
                  const float* X, const float* Wk, float* Y) {
    if (!libxsmm_loader_ok()) return;
    libxsmm_sgemm_f sgemm = (libxsmm_sgemm_f)libxsmm_loader_sym("libxsmm_sgemm");
    if (!sgemm) return;

    int OH = H - KH + 1;
    int OW = W - KW + 1;
    int K  = Cin * KH * KW;
    int N  = OH * OW;

    float* col = (float*)aligned_alloc(64, (size_t)K * N * sizeof(float));
    im2col(Cin, H, W, KH, KW, X, col);

    char trans = 'N';
    int m = N, n = Cout, k = K, lda = N, ldb = K, ldc = N;
    float alpha = 1.0f, beta = 0.0f;
    sgemm(&trans, &trans, &m, &n, &k, &alpha,
          col, &lda, Wk, &ldb, &beta, Y, &ldc);

    free(col);
}
