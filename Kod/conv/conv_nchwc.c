#include "conv.h"
#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>

#define CB 8

extern void conv_packed(int, int, int, int, int, int,
                        const float*, const float*, float*);

static void reorder_X_to_NCHWc(int Cin, int H, int W, const float* X, float* Xt) {
    int Cb_blk = Cin / CB;
    #pragma omp parallel for schedule(static)
    for (int cb = 0; cb < Cb_blk; cb++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                float* dst = &Xt[((cb * H + h) * W + w) * CB];
                for (int c = 0; c < CB; c++)
                    dst[c] = X[((cb * CB + c) * H + h) * W + w];
            }
        }
    }
}

static void reorder_Y_from_NCHWc(int Cout, int OH, int OW, const float* Yt, float* Y) {
    int Cob_blk = Cout / CB;
    #pragma omp parallel for schedule(static)
    for (int cob = 0; cob < Cob_blk; cob++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                const float* src = &Yt[((cob * OH + oh) * OW + ow) * CB];
                for (int c = 0; c < CB; c++)
                    Y[((cob * CB + c) * OH + oh) * OW + ow] = src[c];
            }
        }
    }
}

static void reorder_W_to_blocked(int Cout, int Cin, int KH, int KW,
                                 const float* Wk, float* Wt) {
    int Cib_blk = Cin / CB;
    int Cob_blk = Cout / CB;
    #pragma omp parallel for schedule(static)
    for (int cob = 0; cob < Cob_blk; cob++) {
        for (int cib = 0; cib < Cib_blk; cib++) {
            for (int kh = 0; kh < KH; kh++) {
                for (int kw = 0; kw < KW; kw++) {
                    for (int ci = 0; ci < CB; ci++) {
                        for (int co = 0; co < CB; co++) {
                            int oc = cob * CB + co;
                            int ic = cib * CB + ci;
                            Wt[((((cob * Cib_blk + cib) * KH + kh) * KW + kw) * CB + ci) * CB + co] =
                                Wk[((oc * Cin + ic) * KH + kh) * KW + kw];
                        }
                    }
                }
            }
        }
    }
}

void conv_nchwc(int Cin, int H, int W, int KH, int KW, int Cout,
                const float* X, const float* Wk, float* Y) {
    if ((Cin % CB) != 0 || (Cout % CB) != 0) {
        /* Brak pelnego bloku kanalow (np. RGB, Cin=3): bezposrednie mikrojadro
         * blokowane po kanalach wyjscia (zamiast skromnego conv_packed). */
        conv_oc_blocked(Cin, H, W, KH, KW, Cout, X, Wk, Y);
        return;
    }
    int OH = H - KH + 1;
    int OW = W - KW + 1;
    int Cib_blk = Cin / CB;
    int Cob_blk = Cout / CB;

    float* Xt = (float*)aligned_alloc(64, (size_t)Cin * H * W * sizeof(float));
    float* Wt = (float*)aligned_alloc(64, (size_t)Cout * Cin * KH * KW * sizeof(float));
    float* Yt = (float*)aligned_alloc(64, (size_t)Cout * OH * OW * sizeof(float));

    reorder_X_to_NCHWc(Cin, H, W, X, Xt);
    reorder_W_to_blocked(Cout, Cin, KH, KW, Wk, Wt);

    int ow_blk = (OW / 6) * 6;
    #pragma omp parallel for schedule(static) collapse(2)
    for (int cob = 0; cob < Cob_blk; cob++) {
        for (int oh = 0; oh < OH; oh++) {
            const float* Wt_cob = &Wt[(size_t)cob * Cib_blk * KH * KW * CB * CB];
            float* Yt_cob_oh = &Yt[((size_t)cob * OH + oh) * OW * CB];

            for (int ow = 0; ow < ow_blk; ow += 6) {
                __m256 c0 = _mm256_setzero_ps();
                __m256 c1 = _mm256_setzero_ps();
                __m256 c2 = _mm256_setzero_ps();
                __m256 c3 = _mm256_setzero_ps();
                __m256 c4 = _mm256_setzero_ps();
                __m256 c5 = _mm256_setzero_ps();
                for (int cib = 0; cib < Cib_blk; cib++) {
                    for (int kh = 0; kh < KH; kh++) {
                        for (int kw = 0; kw < KW; kw++) {
                            const float* Xb = &Xt[(((size_t)cib * H + (oh + kh)) * W + (ow + kw)) * CB];
                            const float* Wp = &Wt_cob[((size_t)cib * KH + kh) * KW * CB * CB + (size_t)kw * CB * CB];
                            for (int ci = 0; ci < CB; ci++) {
                                __m256 wv = _mm256_loadu_ps(&Wp[ci * CB]);
                                c0 = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(&Xb[0 * CB + ci]), c0);
                                c1 = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(&Xb[1 * CB + ci]), c1);
                                c2 = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(&Xb[2 * CB + ci]), c2);
                                c3 = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(&Xb[3 * CB + ci]), c3);
                                c4 = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(&Xb[4 * CB + ci]), c4);
                                c5 = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(&Xb[5 * CB + ci]), c5);
                            }
                        }
                    }
                }
                float* yp = &Yt_cob_oh[ow * CB];
                _mm256_storeu_ps(&yp[0 * CB], c0);
                _mm256_storeu_ps(&yp[1 * CB], c1);
                _mm256_storeu_ps(&yp[2 * CB], c2);
                _mm256_storeu_ps(&yp[3 * CB], c3);
                _mm256_storeu_ps(&yp[4 * CB], c4);
                _mm256_storeu_ps(&yp[5 * CB], c5);
            }
            /* Scalar tail for ow not divisible by 6. */
            for (int ow = ow_blk; ow < OW; ow++) {
                float acc[CB] = {0};
                for (int cib = 0; cib < Cib_blk; cib++) {
                    for (int kh = 0; kh < KH; kh++) {
                        for (int kw = 0; kw < KW; kw++) {
                            const float* Xb = &Xt[(((size_t)cib * H + (oh + kh)) * W + (ow + kw)) * CB];
                            const float* Wp = &Wt_cob[((size_t)cib * KH + kh) * KW * CB * CB + (size_t)kw * CB * CB];
                            for (int ci = 0; ci < CB; ci++)
                                for (int co = 0; co < CB; co++)
                                    acc[co] += Wp[ci * CB + co] * Xb[ci];
                        }
                    }
                }
                for (int co = 0; co < CB; co++)
                    Yt_cob_oh[ow * CB + co] = acc[co];
            }
        }
    }

    reorder_Y_from_NCHWc(Cout, OH, OW, Yt, Y);

    free(Xt); free(Wt); free(Yt);
}
