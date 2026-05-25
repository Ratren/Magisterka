#include "conv.h"
#include "conv_internal.h"
#include <omp.h>
#include <stdlib.h>
#include <string.h>

extern void conv_packed_core(int Cin, int H, int W, int KH, int KW, int Cout,
                             int OH, int OW,
                             const float* X, const float* Wk, float* Y,
                             const float* W_pack, float* X_pack);

static inline int imin(int a, int b) { return a < b ? a : b; }
static inline int round_up(int x, int m) { return ((x + m - 1) / m) * m; }

void conv_omp(int Cin, int H, int W, int KH, int KW, int Cout,
              const float* X, const float* Wk, float* Y) {
    int OH = H - KH + 1;
    int OW = W - KW + 1;

    float* W_pack = (float*)aligned_alloc(64, (size_t)Cin * KH * KW * Cout * sizeof(float));
    conv_pack_W_oc_innermost(Cin, KH, KW, Cout, Wk, W_pack);

    #pragma omp parallel
    {
        int nt   = omp_get_num_threads();
        int tid  = omp_get_thread_num();

        int chunk = round_up((Cout + nt - 1) / nt, CONV_MR);
        int oc_start = tid * chunk;
        int oc_end   = imin(Cout, oc_start + chunk);

        if (oc_start < oc_end) {
            int oc_count = oc_end - oc_start;
            float* X_pack = (float*)aligned_alloc(64, (size_t)Cin * KH * KW * CONV_NR * sizeof(float));

            // conv_packed_core expects W_pack laid out as (..., Cout); slice it to this
            // thread's oc range so the core sees oc_count contiguous output channels.
            float* W_pack_local = (float*)aligned_alloc(64, (size_t)Cin * KH * KW * oc_count * sizeof(float));
            for (int ic = 0; ic < Cin; ic++)
                for (int kh = 0; kh < KH; kh++)
                    for (int kw = 0; kw < KW; kw++)
                        for (int j = 0; j < oc_count; j++)
                            W_pack_local[((ic * KH + kh) * KW + kw) * oc_count + j] =
                                W_pack[((ic * KH + kh) * KW + kw) * Cout + oc_start + j];

            conv_packed_core(Cin, H, W, KH, KW, oc_count, OH, OW,
                             X, Wk + (size_t)oc_start * Cin * KH * KW,
                             Y + (size_t)oc_start * OH * OW,
                             W_pack_local, X_pack);

            free(W_pack_local);
            free(X_pack);
        }
    }

    free(W_pack);
}
