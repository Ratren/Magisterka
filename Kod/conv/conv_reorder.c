#include "conv.h"
#include <string.h>

void conv_reorder(int Cin, int H, int W, int KH, int KW, int Cout,
                  const float* X, const float* Wk, float* Y) {
    int OH = H - KH + 1;
    int OW = W - KW + 1;

    memset(Y, 0, (size_t)Cout * OH * OW * sizeof(float));

    for (int oc = 0; oc < Cout; oc++) {
        for (int ic = 0; ic < Cin; ic++) {
            for (int kh = 0; kh < KH; kh++) {
                for (int kw = 0; kw < KW; kw++) {
                    float w = Wk[oc * Cin * KH * KW + ic * KH * KW +
                                 (KH - 1 - kh) * KW + (KW - 1 - kw)];
                    for (int oh = 0; oh < OH; oh++) {
                        const float* xrow = &X[ic * H * W + (oh + kh) * W + kw];
                        float* yrow = &Y[oc * OH * OW + oh * OW];
                        for (int ow = 0; ow < OW; ow++) {
                            yrow[ow] += w * xrow[ow];
                        }
                    }
                }
            }
        }
    }
}
