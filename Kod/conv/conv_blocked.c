#include "conv.h"
#include <string.h>

#define BLOCK_OC 32
#define BLOCK_OH 16
#define BLOCK_IC 32

static inline int imin(int a, int b) { return a < b ? a : b; }

void conv_blocked(int Cin, int H, int W, int KH, int KW, int Cout,
                  const float* X, const float* Wk, float* Y) {
    int OH = H - KH + 1;
    int OW = W - KW + 1;

    memset(Y, 0, (size_t)Cout * OH * OW * sizeof(float));

    for (int oc0 = 0; oc0 < Cout; oc0 += BLOCK_OC) {
        int oc_end = imin(oc0 + BLOCK_OC, Cout);
        for (int oh0 = 0; oh0 < OH; oh0 += BLOCK_OH) {
            int oh_end = imin(oh0 + BLOCK_OH, OH);
            for (int ic0 = 0; ic0 < Cin; ic0 += BLOCK_IC) {
                int ic_end = imin(ic0 + BLOCK_IC, Cin);
                for (int oc = oc0; oc < oc_end; oc++) {
                    for (int ic = ic0; ic < ic_end; ic++) {
                        for (int kh = 0; kh < KH; kh++) {
                            for (int kw = 0; kw < KW; kw++) {
                                float w = Wk[oc * Cin * KH * KW + ic * KH * KW + kh * KW + kw];
                                for (int oh = oh0; oh < oh_end; oh++) {
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
        }
    }
}
