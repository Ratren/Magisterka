#include "conv.h"
#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>

/* Mikrojadro bezposrednie dla malej liczby kanalow wejscia (np. obraz RGB,
 * Cin=3), gdzie NCHWc8 nie tworzy pelnego bloku kanalow wejscia, a Winograd
 * degeneruje sie do mnozen o wymiarze wspolnym K=Cin. Zamiast blokowac kanaly
 * WEJSCIA, blokujemy kanaly WYJSCIA: osiem sasiednich kanalow wyjscia wpada do
 * jednego rejestru YMM (CB=8 liczb pojedynczej precyzji), a kanaly wejscia sa
 * zwykla, krotka petla redukcji. Wejscie pozostaje w ukladzie NCHW -- wartosci
 * wejscia sa tylko powielane (broadcast). Wagi i wynik blokuje sie po kanale
 * wyjscia; tani reorder wyniku do NCHW wykonuje sie na koncu.
 *
 * Gorace mikrojadro liczy kafelek 8 kanalow wyjscia x 6 pozycji ow, utrzymujac
 * szesc akumulatorow. W kazdym kroku po (ic, kh, kw): jeden odczyt 8 wag,
 * szesc powielen wartosci wejscia i szesc instrukcji FMA -- osiem zywych
 * rejestrow YMM, wiec bez spillingu i z zapasem dla planisty out-of-order. */

#define OCB 8   /* blok kanalow wyjscia = szerokosc rejestru YMM (float)   */
#define OWB 6   /* kafelek pozycji ow (szesc akumulatorow w mikrojadrze)    */

/* Wagi Wk[oc][ic][kh][kw] -> Wt[ocb][ic][kh][kw][8 oc] (8 oc ciagle). */
static void reorder_W_ocblocked(int Cout, int Cin, int KH, int KW,
                                const float* Wk, float* Wt) {
    int Cob = Cout / OCB;
    for (int cob = 0; cob < Cob; cob++)
        for (int ic = 0; ic < Cin; ic++)
            for (int kh = 0; kh < KH; kh++)
                for (int kw = 0; kw < KW; kw++)
                    for (int co = 0; co < OCB; co++) {
                        int oc = cob * OCB + co;
                        Wt[((((size_t)cob * Cin + ic) * KH + kh) * KW + kw) * OCB + co] =
                            Wk[(((size_t)oc * Cin + ic) * KH + kh) * KW + kw];
                    }
}

/* Yt[ocb][oh][ow][8] -> Y[oc][oh][ow] (NCHW). */
static void reorder_Y_from_ocblocked(int Cout, int OH, int OW,
                                     const float* Yt, float* Y) {
    int Cob = Cout / OCB;
    #pragma omp parallel for schedule(static)
    for (int cob = 0; cob < Cob; cob++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++) {
                const float* src = &Yt[(((size_t)cob * OH + oh) * OW + ow) * OCB];
                for (int co = 0; co < OCB; co++)
                    Y[(((size_t)(cob * OCB + co) * OH + oh) * OW + ow)] = src[co];
            }
}

void conv_oc_blocked(int Cin, int H, int W, int KH, int KW, int Cout,
                     const float* X, const float* Wk, float* Y) {
    int OH = H - KH + 1;
    int OW = W - KW + 1;
    int Cob = Cout / OCB;
    int ow_blk = (OW / OWB) * OWB;

    float* Wt = (float*)aligned_alloc(64, (size_t)Cob * Cin * KH * KW * OCB * sizeof(float));
    float* Yt = (float*)aligned_alloc(64, (size_t)Cob * OH * OW * OCB * sizeof(float));
    reorder_W_ocblocked(Cout, Cin, KH, KW, Wk, Wt);

    #pragma omp parallel for schedule(static) collapse(2)
    for (int cob = 0; cob < Cob; cob++) {
        for (int oh = 0; oh < OH; oh++) {
            const float* Wt_cob = &Wt[(size_t)cob * Cin * KH * KW * OCB];
            float* Yt_co = &Yt[((size_t)cob * OH + oh) * OW * OCB];

            for (int ow = 0; ow < ow_blk; ow += OWB) {
                __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
                __m256 c2 = _mm256_setzero_ps(), c3 = _mm256_setzero_ps();
                __m256 c4 = _mm256_setzero_ps(), c5 = _mm256_setzero_ps();
                for (int ic = 0; ic < Cin; ic++) {
                    for (int kh = 0; kh < KH; kh++) {
                        const float* xr = &X[((size_t)ic * H + (oh + kh)) * W + ow];
                        const float* wbase = &Wt_cob[(((size_t)ic * KH + kh) * KW) * OCB];
                        for (int kw = 0; kw < KW; kw++) {
                            __m256 wv = _mm256_load_ps(&wbase[kw * OCB]);
                            const float* xk = &xr[kw];
                            c0 = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(&xk[0]), c0);
                            c1 = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(&xk[1]), c1);
                            c2 = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(&xk[2]), c2);
                            c3 = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(&xk[3]), c3);
                            c4 = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(&xk[4]), c4);
                            c5 = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(&xk[5]), c5);
                        }
                    }
                }
                float* yp = &Yt_co[ow * OCB];
                _mm256_store_ps(&yp[0 * OCB], c0);
                _mm256_store_ps(&yp[1 * OCB], c1);
                _mm256_store_ps(&yp[2 * OCB], c2);
                _mm256_store_ps(&yp[3 * OCB], c3);
                _mm256_store_ps(&yp[4 * OCB], c4);
                _mm256_store_ps(&yp[5 * OCB], c5);
            }
            /* Reszta pozycji ow (gdy OW nie jest wielokrotnoscia OWB). */
            for (int ow = ow_blk; ow < OW; ow++) {
                float acc[OCB] = {0};
                for (int ic = 0; ic < Cin; ic++)
                    for (int kh = 0; kh < KH; kh++)
                        for (int kw = 0; kw < KW; kw++) {
                            const float* wp = &Wt_cob[(((size_t)ic * KH + kh) * KW + kw) * OCB];
                            float xv = X[((size_t)ic * H + (oh + kh)) * W + (ow + kw)];
                            for (int co = 0; co < OCB; co++)
                                acc[co] += wp[co] * xv;
                        }
                for (int co = 0; co < OCB; co++)
                    Yt_co[ow * OCB + co] = acc[co];
            }
        }
    }

    reorder_Y_from_ocblocked(Cout, OH, OW, Yt, Y);

    /* Reszta kanalow wyjscia (gdy Cout nie jest wielokrotnoscia OCB) -- droga
     * skalarna wprost do Y; w presetach RGB Cout=64/32, wiec sie nie uruchamia. */
    for (int oc = Cob * OCB; oc < Cout; oc++) {
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++) {
                float s = 0.0f;
                for (int ic = 0; ic < Cin; ic++)
                    for (int kh = 0; kh < KH; kh++)
                        for (int kw = 0; kw < KW; kw++)
                            s += X[((size_t)ic * H + (oh + kh)) * W + (ow + kw)] *
                                 Wk[(((size_t)oc * Cin + ic) * KH + kh) * KW + kw];
                Y[((size_t)oc * OH + oh) * OW + ow] = s;
            }
    }

    free(Wt);
    free(Yt);
}
