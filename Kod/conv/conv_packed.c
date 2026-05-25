#include "conv.h"
#include "conv_internal.h"
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

#define ALWAYS_INLINE static inline __attribute__((always_inline))
#define HOT __attribute__((hot))

static inline int imin(int a, int b) { return a < b ? a : b; }

static void pack_W(int Cin, int KH, int KW, int Cout,
                   const float* Wk, float* W_pack) {
    for (int ic = 0; ic < Cin; ic++)
        for (int kh = 0; kh < KH; kh++)
            for (int kw = 0; kw < KW; kw++)
                for (int oc = 0; oc < Cout; oc++)
                    W_pack[((ic * KH + kh) * KW + kw) * Cout + oc] =
                        Wk[((oc * Cin + ic) * KH + kh) * KW + kw];
}

ALWAYS_INLINE HOT void ukr_packed(int Cin, int KH, int KW, int Cout,
                                  int oc, int OH, int OW, int oh, int ow,
                                  const float* X_pack, const float* W_pack,
                                  float* Y) {
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();

    int steps = Cin * KH * KW;
    for (int p = 0; p < steps; p++) {
        __m256 x0 = _mm256_load_ps(&X_pack[p * CONV_NR + 0]);
        __m256 x1 = _mm256_load_ps(&X_pack[p * CONV_NR + 8]);
        const float* wp = &W_pack[p * Cout + oc];
        __m256 w;
        w = _mm256_broadcast_ss(&wp[0]); c00 = _mm256_fmadd_ps(w, x0, c00); c01 = _mm256_fmadd_ps(w, x1, c01);
        w = _mm256_broadcast_ss(&wp[1]); c10 = _mm256_fmadd_ps(w, x0, c10); c11 = _mm256_fmadd_ps(w, x1, c11);
        w = _mm256_broadcast_ss(&wp[2]); c20 = _mm256_fmadd_ps(w, x0, c20); c21 = _mm256_fmadd_ps(w, x1, c21);
        w = _mm256_broadcast_ss(&wp[3]); c30 = _mm256_fmadd_ps(w, x0, c30); c31 = _mm256_fmadd_ps(w, x1, c31);
        w = _mm256_broadcast_ss(&wp[4]); c40 = _mm256_fmadd_ps(w, x0, c40); c41 = _mm256_fmadd_ps(w, x1, c41);
        w = _mm256_broadcast_ss(&wp[5]); c50 = _mm256_fmadd_ps(w, x0, c50); c51 = _mm256_fmadd_ps(w, x1, c51);
    }

    float* yp = &Y[oc * OH * OW + oh * OW + ow];
    _mm256_storeu_ps(yp + 0 * OH * OW + 0, c00); _mm256_storeu_ps(yp + 0 * OH * OW + 8, c01);
    _mm256_storeu_ps(yp + 1 * OH * OW + 0, c10); _mm256_storeu_ps(yp + 1 * OH * OW + 8, c11);
    _mm256_storeu_ps(yp + 2 * OH * OW + 0, c20); _mm256_storeu_ps(yp + 2 * OH * OW + 8, c21);
    _mm256_storeu_ps(yp + 3 * OH * OW + 0, c30); _mm256_storeu_ps(yp + 3 * OH * OW + 8, c31);
    _mm256_storeu_ps(yp + 4 * OH * OW + 0, c40); _mm256_storeu_ps(yp + 4 * OH * OW + 8, c41);
    _mm256_storeu_ps(yp + 5 * OH * OW + 0, c50); _mm256_storeu_ps(yp + 5 * OH * OW + 8, c51);
}

static void edge_scalar_p(int Cin, int H, int W, int KH, int KW, int Cout,
                          int oc_lo, int oc_hi, int oh, int ow_lo, int ow_hi,
                          int OH, int OW,
                          const float* X, const float* Wk, float* Y) {
    for (int oc = oc_lo; oc < oc_hi; oc++) {
        for (int ow = ow_lo; ow < ow_hi; ow++) {
            float sum = 0.0f;
            for (int ic = 0; ic < Cin; ic++)
                for (int kh = 0; kh < KH; kh++)
                    for (int kw = 0; kw < KW; kw++)
                        sum += X[ic * H * W + (oh + kh) * W + (ow + kw)] *
                               Wk[((oc * Cin + ic) * KH + kh) * KW + kw];
            Y[oc * OH * OW + oh * OW + ow] = sum;
        }
    }
}

void conv_packed_core(int Cin, int H, int W, int KH, int KW, int Cout,
                      int OH, int OW,
                      const float* X, const float* Wk, float* Y,
                      const float* W_pack, float* X_pack) {
    int oc_blk = (Cout / CONV_MR) * CONV_MR;
    int ow_blk = (OW / CONV_NR) * CONV_NR;
    int steps  = Cin * KH * KW;

    for (int oh = 0; oh < OH; oh++) {
        for (int ow = 0; ow < ow_blk; ow += CONV_NR) {
            for (int ic = 0; ic < Cin; ic++) {
                for (int kh = 0; kh < KH; kh++) {
                    for (int kw = 0; kw < KW; kw++) {
                        const float* xrow = &X[ic * H * W + (oh + kh) * W + (ow + kw)];
                        int p = (ic * KH + kh) * KW + kw;
                        memcpy(&X_pack[p * CONV_NR], xrow, CONV_NR * sizeof(float));
                    }
                }
            }
            (void)steps;
            for (int oc = 0; oc < oc_blk; oc += CONV_MR) {
                ukr_packed(Cin, KH, KW, Cout, oc, OH, OW, oh, ow,
                           X_pack, W_pack, Y);
            }
            if (oc_blk < Cout) {
                edge_scalar_p(Cin, H, W, KH, KW, Cout,
                              oc_blk, Cout, oh, ow, ow + CONV_NR,
                              OH, OW, X, Wk, Y);
            }
        }
        if (ow_blk < OW) {
            edge_scalar_p(Cin, H, W, KH, KW, Cout,
                          0, Cout, oh, ow_blk, OW,
                          OH, OW, X, Wk, Y);
        }
    }
}

HOT void conv_packed(int Cin, int H, int W, int KH, int KW, int Cout,
                     const float* X, const float* Wk, float* Y) {
    int OH = H - KH + 1;
    int OW = W - KW + 1;

    float* W_pack = (float*)aligned_alloc(64, (size_t)Cin * KH * KW * Cout * sizeof(float));
    float* X_pack = (float*)aligned_alloc(64, (size_t)Cin * KH * KW * CONV_NR * sizeof(float));
    pack_W(Cin, KH, KW, Cout, Wk, W_pack);

    conv_packed_core(Cin, H, W, KH, KW, Cout, OH, OW, X, Wk, Y, W_pack, X_pack);

    free(X_pack);
    free(W_pack);
    (void)imin;
}
