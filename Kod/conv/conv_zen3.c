#include "conv.h"

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
    conv_zen3(Cin, H, W, KH, KW, Cout, X, Wk, Y);
}
