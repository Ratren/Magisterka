#include "conv.h"
#include "common.h"
#include <stdlib.h>
#include <string.h>

/* libxsmm conv wrapper. Two layers of fallback:
   - If libxsmm isn't loaded at all (loader returned 0), this is a no-op
     so the benchmark slot shows N/A.
   - If libxsmm is loaded, we use its public sgemm entry (libxsmm_sgemm)
     after an im2col, which is the simplest reliable cross-version API.
     A future improvement could call libxsmm_dnn_create_conv_op directly
     to skip the im2col; the API there has changed across releases so we
     stick with sgemm for portability. */

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

    /* Row-major Y(Cout, N) = Wk(Cout, K) * col(K, N). libxsmm_sgemm is
       Fortran (column-major), same trick as conv_im2col_blis: view both
       operands transposed and call (N, Cout, K) * (K) * (Cout). */
    char trans = 'N';
    int m = N, n = Cout, k = K, lda = N, ldb = K, ldc = N;
    float alpha = 1.0f, beta = 0.0f;
    sgemm(&trans, &trans, &m, &n, &k, &alpha,
          col, &lda, Wk, &ldb, &beta, Y, &ldc);

    free(col);
}
