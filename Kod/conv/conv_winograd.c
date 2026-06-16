#include "conv.h"
#include "openblas/cblas.h"
#include <omp.h>
#include <stdlib.h>
#include <string.h>

extern void conv_packed(int, int, int, int, int, int,
                        const float*, const float*, float*);

#define TT 16  /* number of tile positions = 4x4 */

/* Filter transform: u = G g G^T for one 3x3 filter g, output 4x4 matrix u. */
static inline void wino_filter_transform(const float* g, float* u) {
    /* tmp = G g, then u = tmp G^T */
    float tmp[4 * 3];
    for (int j = 0; j < 3; j++) {
        float g0j = g[0 * 3 + j], g1j = g[1 * 3 + j], g2j = g[2 * 3 + j];
        tmp[0 * 3 + j] = g0j;
        tmp[1 * 3 + j] = 0.5f * (g0j + g1j + g2j);
        tmp[2 * 3 + j] = 0.5f * (g0j - g1j + g2j);
        tmp[3 * 3 + j] = g2j;
    }
    for (int i = 0; i < 4; i++) {
        float t0 = tmp[i * 3 + 0], t1 = tmp[i * 3 + 1], t2 = tmp[i * 3 + 2];
        u[i * 4 + 0] = t0;
        u[i * 4 + 1] = 0.5f * (t0 + t1 + t2);
        u[i * 4 + 2] = 0.5f * (t0 - t1 + t2);
        u[i * 4 + 3] = t2;
    }
}

static inline void wino_input_transform(const float d[4][4], float v[4][4]) {
    float t[4][4];
    /* tmp = B^T d */
    for (int j = 0; j < 4; j++) {
        t[0][j] = d[0][j] - d[2][j];
        t[1][j] = d[1][j] + d[2][j];
        t[2][j] = -d[1][j] + d[2][j];
        t[3][j] = d[1][j] - d[3][j];
    }
    /* v = tmp B */
    for (int i = 0; i < 4; i++) {
        v[i][0] = t[i][0] - t[i][2];
        v[i][1] = t[i][1] + t[i][2];
        v[i][2] = -t[i][1] + t[i][2];
        v[i][3] = t[i][1] - t[i][3];
    }
}

/* Output transform: Y_2x2 = A^T M A for one 4x4 tile M. */
static inline void wino_output_transform(const float m[4][4], float y[2][2]) {
    float t[2][4];
    for (int j = 0; j < 4; j++) {
        t[0][j] = m[0][j] + m[1][j] + m[2][j];
        t[1][j] = m[1][j] - m[2][j] - m[3][j];
    }
    for (int i = 0; i < 2; i++) {
        y[i][0] = t[i][0] + t[i][1] + t[i][2];
        y[i][1] = t[i][1] - t[i][2] - t[i][3];
    }
}

void conv_winograd(int Cin, int H, int W, int KH, int KW, int Cout,
                   const float* X, const float* Wk, float* Y) {
    if (KH != 3 || KW != 3) {
        conv_packed(Cin, H, W, KH, KW, Cout, X, Wk, Y);
        return;
    }
    int OH = H - 2, OW = W - 2;
    int Th = OH / 2, Tw = OW / 2;
    if (Th == 0 || Tw == 0) {
        conv_packed(Cin, H, W, KH, KW, Cout, X, Wk, Y);
        return;
    }
    int T = Th * Tw;

    size_t U_bytes = (size_t)TT * Cout * Cin * sizeof(float);
    float* U = (float*)aligned_alloc(64, U_bytes);
    #pragma omp parallel for schedule(static) collapse(2)
    for (int oc = 0; oc < Cout; oc++) {
        for (int ic = 0; ic < Cin; ic++) {
            const float* g = &Wk[(oc * Cin + ic) * 9];
            float gf[9];
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    gf[i * 3 + j] = g[(2 - i) * 3 + (2 - j)];
            float u[16];
            wino_filter_transform(gf, u);
            for (int i = 0; i < TT; i++)
                U[(size_t)i * Cout * Cin + (size_t)oc * Cin + ic] = u[i];
        }
    }

    size_t V_bytes = (size_t)TT * Cin * T * sizeof(float);
    float* V = (float*)aligned_alloc(64, V_bytes);
    #pragma omp parallel for schedule(static)
    for (int ic = 0; ic < Cin; ic++) {
        const float* Xc = &X[(size_t)ic * H * W];
        for (int th = 0; th < Th; th++) {
            for (int tw = 0; tw < Tw; tw++) {
                int h0 = th * 2, w0 = tw * 2;
                float d[4][4];
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 4; j++)
                        d[i][j] = Xc[(h0 + i) * W + (w0 + j)];
                float v[4][4];
                wino_input_transform(d, v);
                int t_idx = th * Tw + tw;
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 4; j++)
                        V[(size_t)(i * 4 + j) * Cin * T + (size_t)ic * T + t_idx] = v[i][j];
            }
        }
    }

    size_t M_bytes = (size_t)TT * Cout * T * sizeof(float);
    float* M = (float*)aligned_alloc(64, M_bytes);
    for (int i = 0; i < TT; i++) {
        const float* Ui = &U[(size_t)i * Cout * Cin];
        const float* Vi = &V[(size_t)i * Cin * T];
        float* Mi = &M[(size_t)i * Cout * T];
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    Cout, T, Cin, 1.0f, Ui, Cin, Vi, T, 0.0f, Mi, T);
    }

    #pragma omp parallel for schedule(static) collapse(2)
    for (int oc = 0; oc < Cout; oc++) {
        for (int th = 0; th < Th; th++) {
            for (int tw = 0; tw < Tw; tw++) {
                int t_idx = th * Tw + tw;
                float m[4][4];
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 4; j++)
                        m[i][j] = M[(size_t)(i * 4 + j) * Cout * T + (size_t)oc * T + t_idx];
                float y[2][2];
                wino_output_transform(m, y);
                int oh = th * 2, ow = tw * 2;
                float* Yc = &Y[(size_t)oc * OH * OW];
                Yc[(oh + 0) * OW + (ow + 0)] = y[0][0];
                Yc[(oh + 0) * OW + (ow + 1)] = y[0][1];
                Yc[(oh + 1) * OW + (ow + 0)] = y[1][0];
                Yc[(oh + 1) * OW + (ow + 1)] = y[1][1];
            }
        }
    }

    if ((OH % 2) || (OW % 2)) {
        for (int oc = 0; oc < Cout; oc++) {
            for (int oh = Th * 2; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    float s = 0.0f;
                    for (int ic = 0; ic < Cin; ic++)
                        for (int kh = 0; kh < 3; kh++)
                            for (int kw = 0; kw < 3; kw++)
                                s += X[((size_t)ic * H + (oh + kh)) * W + (ow + kw)] *
                                     Wk[((size_t)oc * Cin + ic) * 9 + (2 - kh) * 3 + (2 - kw)];
                    Y[((size_t)oc * OH + oh) * OW + ow] = s;
                }
            }
            for (int oh = 0; oh < Th * 2; oh++) {
                for (int ow = Tw * 2; ow < OW; ow++) {
                    float s = 0.0f;
                    for (int ic = 0; ic < Cin; ic++)
                        for (int kh = 0; kh < 3; kh++)
                            for (int kw = 0; kw < 3; kw++)
                                s += X[((size_t)ic * H + (oh + kh)) * W + (ow + kw)] *
                                     Wk[((size_t)oc * Cin + ic) * 9 + (2 - kh) * 3 + (2 - kw)];
                    Y[((size_t)oc * OH + oh) * OW + ow] = s;
                }
            }
        }
    }

    free(U); free(V); free(M);
}
