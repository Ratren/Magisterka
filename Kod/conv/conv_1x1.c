#include "conv.h"
#include "openblas/cblas.h"

extern void conv_packed(int, int, int, int, int, int,
                        const float*, const float*, float*);

void conv_1x1(int Cin, int H, int W, int KH, int KW, int Cout,
              const float* X, const float* Wk, float* Y) {
    if (KH != 1 || KW != 1) {
        conv_packed(Cin, H, W, KH, KW, Cout, X, Wk, Y);
        return;
    }
    int N = H * W;  /* OH = H, OW = W since KH=KW=1, no padding, stride 1 */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                Cout, N, Cin,
                1.0f, Wk, Cin, X, N, 0.0f, Y, N);
}
