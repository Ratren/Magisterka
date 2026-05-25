#include "conv.h"
#include "conv_internal.h"
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

#define ALWAYS_INLINE static inline __attribute__((always_inline))
#define HOT __attribute__((hot))

// Pack Wk from (OC, IC, KH, KW) to (IC, KH, KW, OC).
static void pack_W_oc_innermost(int Cin, int KH, int KW, int Cout,
                                const float* Wk, float* W_pack) {
    for (int ic = 0; ic < Cin; ic++) {
        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                for (int oc = 0; oc < Cout; oc++) {
                    W_pack[((ic * KH + kh) * KW + kw) * Cout + oc] =
                        Wk[((oc * Cin + ic) * KH + kh) * KW + kw];
                }
            }
        }
    }
}

ALWAYS_INLINE HOT void ukr_6oc_16ow(int Cin, int KH, int KW,
                                    int H, int W, int OH, int OW,
                                    int oc, int oh, int ow,
                                    const float* X, const float* W_pack, int Cout,
                                    float* Y) {
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();

    for (int ic = 0; ic < Cin; ic++) {
        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                const float* xrow = &X[ic * H * W + (oh + kh) * W + (ow + kw)];
                __m256 x0 = _mm256_loadu_ps(xrow + 0);
                __m256 x1 = _mm256_loadu_ps(xrow + 8);

                const float* wp = &W_pack[((ic * KH + kh) * KW + kw) * Cout + oc];
                __m256 w;
                w = _mm256_broadcast_ss(&wp[0]); c00 = _mm256_fmadd_ps(w, x0, c00); c01 = _mm256_fmadd_ps(w, x1, c01);
                w = _mm256_broadcast_ss(&wp[1]); c10 = _mm256_fmadd_ps(w, x0, c10); c11 = _mm256_fmadd_ps(w, x1, c11);
                w = _mm256_broadcast_ss(&wp[2]); c20 = _mm256_fmadd_ps(w, x0, c20); c21 = _mm256_fmadd_ps(w, x1, c21);
                w = _mm256_broadcast_ss(&wp[3]); c30 = _mm256_fmadd_ps(w, x0, c30); c31 = _mm256_fmadd_ps(w, x1, c31);
                w = _mm256_broadcast_ss(&wp[4]); c40 = _mm256_fmadd_ps(w, x0, c40); c41 = _mm256_fmadd_ps(w, x1, c41);
                w = _mm256_broadcast_ss(&wp[5]); c50 = _mm256_fmadd_ps(w, x0, c50); c51 = _mm256_fmadd_ps(w, x1, c51);
            }
        }
    }

    float* yp = &Y[oc * OH * OW + oh * OW + ow];
    _mm256_storeu_ps(yp + 0 * OH * OW + 0, c00); _mm256_storeu_ps(yp + 0 * OH * OW + 8, c01);
    _mm256_storeu_ps(yp + 1 * OH * OW + 0, c10); _mm256_storeu_ps(yp + 1 * OH * OW + 8, c11);
    _mm256_storeu_ps(yp + 2 * OH * OW + 0, c20); _mm256_storeu_ps(yp + 2 * OH * OW + 8, c21);
    _mm256_storeu_ps(yp + 3 * OH * OW + 0, c30); _mm256_storeu_ps(yp + 3 * OH * OW + 8, c31);
    _mm256_storeu_ps(yp + 4 * OH * OW + 0, c40); _mm256_storeu_ps(yp + 4 * OH * OW + 8, c41);
    _mm256_storeu_ps(yp + 5 * OH * OW + 0, c50); _mm256_storeu_ps(yp + 5 * OH * OW + 8, c51);
}

static void edge_scalar(int Cin, int H, int W, int KH, int KW, int Cout,
                        int oc_lo, int oc_hi, int oh, int ow_lo, int ow_hi,
                        int OH, int OW,
                        const float* X, const float* Wk, float* Y) {
    for (int oc = oc_lo; oc < oc_hi; oc++) {
        for (int ow = ow_lo; ow < ow_hi; ow++) {
            float sum = 0.0f;
            for (int ic = 0; ic < Cin; ic++) {
                for (int kh = 0; kh < KH; kh++) {
                    for (int kw = 0; kw < KW; kw++) {
                        sum += X[ic * H * W + (oh + kh) * W + (ow + kw)] *
                               Wk[((oc * Cin + ic) * KH + kh) * KW + kw];
                    }
                }
            }
            Y[oc * OH * OW + oh * OW + ow] = sum;
        }
    }
}

HOT void conv_microkernel(int Cin, int H, int W, int KH, int KW, int Cout,
                          const float* X, const float* Wk, float* Y) {
    int OH = H - KH + 1;
    int OW = W - KW + 1;

    float* W_pack = (float*)aligned_alloc(64, (size_t)Cin * KH * KW * Cout * sizeof(float));
    pack_W_oc_innermost(Cin, KH, KW, Cout, Wk, W_pack);

    int oc_blk = (Cout / CONV_MR) * CONV_MR;
    int ow_blk = (OW / CONV_NR) * CONV_NR;

    for (int oc = 0; oc < oc_blk; oc += CONV_MR) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < ow_blk; ow += CONV_NR) {
                ukr_6oc_16ow(Cin, KH, KW, H, W, OH, OW,
                             oc, oh, ow, X, W_pack, Cout, Y);
            }
            if (ow_blk < OW) {
                edge_scalar(Cin, H, W, KH, KW, Cout,
                            oc, oc + CONV_MR, oh, ow_blk, OW,
                            OH, OW, X, Wk, Y);
            }
        }
    }
    if (oc_blk < Cout) {
        for (int oh = 0; oh < OH; oh++) {
            edge_scalar(Cin, H, W, KH, KW, Cout,
                        oc_blk, Cout, oh, 0, OW,
                        OH, OW, X, Wk, Y);
        }
    }

    free(W_pack);
}
