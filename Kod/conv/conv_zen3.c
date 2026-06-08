#include "conv.h"

void conv_zen3(int Cin, int H, int W, int KH, int KW, int Cout,
               const float* X, const float* Wk, float* Y) {
    if (KH == 1 && KW == 1) {
        conv_1x1(Cin, H, W, KH, KW, Cout, X, Wk, Y);
        return;
    }
    /* Maly Cin (np. obraz RGB, Cin=3): NCHWc8 nie tworzy pelnego bloku kanalow
     * wejscia, a 16 mnozen macierzy Winograda ma wymiar wspolny K=Cin (chude,
     * bez reuzycia po K). Kierujemy na bezposrednie mikrojadro blokowane po
     * kanalach WYJSCIA, ktore nie wymaga ani blokowania kanalow wejscia, ani
     * dopelniania ich zerami. */
    if (Cin < 8) {
        conv_oc_blocked(Cin, H, W, KH, KW, Cout, X, Wk, Y);
        return;
    }
    int OH = H - KH + 1;
    int OW = W - KW + 1;
    if (KH == 3 && KW == 3 && (OH % 2 == 0) && (OW % 2 == 0)) {
        /* Winograd oplaca sie tylko, gdy jego 16 wsadowych mnozen macierzy jest
         * dostatecznie duzych: wymiar N to liczba kafelkow T = (OH/2)(OW/2),
         * a M = Cout, K = Cin. Przy malej liczbie kafelkow (male mapy, np.
         * 14x14 -> T=36) albo malej liczbie kanalow (np. Cin=Cout=16) mnozenia
         * robia sie chude, a narzut transformat dominuje -- wtedy bezposrednie
         * NCHWc8 jest szybsze (zmierzone: tiny, large, xlarge). Progi dobrano
         * empirycznie z krzywej skrzyzowania Winograd/NCHWc8 na Zen 3. */
        int tiles  = (OH / 2) * (OW / 2);
        int min_ch = (Cin < Cout) ? Cin : Cout;
        if (tiles >= 64 && min_ch >= 32) {
            conv_winograd(Cin, H, W, KH, KW, Cout, X, Wk, Y);
            return;
        }
    }
    conv_nchwc(Cin, H, W, KH, KW, Cout, X, Wk, Y);
}

void conv_zen3_omp(int Cin, int H, int W, int KH, int KW, int Cout,
                   const float* X, const float* Wk, float* Y) {
    conv_zen3(Cin, H, W, KH, KW, Cout, X, Wk, Y);
}
