#include "conv.h"

void conv_naive(int Cin, int H, int W, int KH, int KW, int Cout,
                const float* X, const float* Wk, float* Y) {
    int OH = H - KH + 1;
    int OW = W - KW + 1;
    for (int oc = 0; oc < Cout; oc++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                float sum = 0.0f;
                for (int ic = 0; ic < Cin; ic++) {
                    for (int kh = 0; kh < KH; kh++) {
                        for (int kw = 0; kw < KW; kw++) {
                            sum += X[ic * H * W + (oh + kh) * W + (ow + kw)] *
                                   Wk[oc * Cin * KH * KW + ic * KH * KW +
                                      (KH - 1 - kh) * KW + (KW - 1 - kw)];
                        }
                    }
                }
                Y[oc * OH * OW + oh * OW + ow] = sum;
            }
        }
    }
}
