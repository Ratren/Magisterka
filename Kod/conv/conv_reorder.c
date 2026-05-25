#include "conv.h"
#include <string.h>

// Loop order: oc, ic, kh, kw, oh, ow.
// For fixed (oc, ic, kh, kw), the inner (oh, ow) loops read X[ic, oh+kh, ow+kw]
// sequentially in ow and write Y[oc, oh, ow] sequentially in ow.
// Wk[oc, ic, kh, kw] is loop-invariant in the inner loops -- hoisted out.
void conv_reorder(int Cin, int H, int W, int KH, int KW, int Cout,
                  const float* X, const float* Wk, float* Y) {
    int OH = H - KH + 1;
    int OW = W - KW + 1;

    memset(Y, 0, (size_t)Cout * OH * OW * sizeof(float));

    for (int oc = 0; oc < Cout; oc++) {
        for (int ic = 0; ic < Cin; ic++) {
            for (int kh = 0; kh < KH; kh++) {
                for (int kw = 0; kw < KW; kw++) {
                    float w = Wk[oc * Cin * KH * KW + ic * KH * KW + kh * KW + kw];
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
