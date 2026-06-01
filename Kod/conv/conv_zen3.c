#include "conv.h"

/* Top-level dispatcher: pick the right specialised path based on kernel
   shape, mirroring how gemm_zen3_best_omp dispatches by size. The
   single-thread and multi-thread variants share the same dispatch tree;
   the called functions take care of their own OMP scheduling. */

void conv_zen3(int Cin, int H, int W, int KH, int KW, int Cout,
               const float* X, const float* Wk, float* Y) {
    if (KH == 1 && KW == 1) {
        conv_1x1(Cin, H, W, KH, KW, Cout, X, Wk, Y);
        return;
    }
    int OH = H - KH + 1;
    int OW = W - KW + 1;
    if (KH == 3 && KW == 3 && OH * OW >= 16 && (OH % 2 == 0) && (OW % 2 == 0)) {
        conv_winograd(Cin, H, W, KH, KW, Cout, X, Wk, Y);
        return;
    }
    conv_nchwc(Cin, H, W, KH, KW, Cout, X, Wk, Y);
}

void conv_zen3_omp(int Cin, int H, int W, int KH, int KW, int Cout,
                   const float* X, const float* Wk, float* Y) {
    /* All the specialised paths already use #pragma omp parallel for in
       their layout transforms and main loops. The top-level OMP region
       lives inside the dispatched function rather than around the
       dispatch itself - same model as gemm_zen3_best_omp. */
    conv_zen3(Cin, H, W, KH, KW, Cout, X, Wk, Y);
}
