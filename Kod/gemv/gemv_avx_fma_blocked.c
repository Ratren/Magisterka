#include "gemv.h"
#include <immintrin.h>
#include <string.h>
#include <stdint.h>

// ============================================================
// Atrybuty kompilacji
// ============================================================

#define ALWAYS_INLINE static inline __attribute__((always_inline))
#define HOT __attribute__((hot))
#define ASSUME_ALIGNED(ptr, align) __builtin_assume_aligned(ptr, align)

// ============================================================
// Parametry blokowania jako stale kompilacji (constexpr-style)
// Uzycie enum wymusza traktowanie jako stale compile-time
// ============================================================

enum L1Params {
    L1_ROW_BLOCK = 64,
    L1_COL_BLOCK = 512
};

enum L2Params {
    L2_ROW_BLOCK = 128,
    L2_COL_BLOCK = 256
};

enum L3Params {
    L3_ROW_BLOCK = 64,
    L3_COL_BLOCK = 512
};

enum CacheThresholds {
    L1_THRESHOLD = 24 * 1024,   // 75% z 32KB
    L2_THRESHOLD = 384 * 1024   // 75% z 512KB
};

enum PrefetchDist {
    PREFETCH_L2_DIST = 128
};

// ============================================================
// Redukcja horyzontalna __m256d -> double
// ============================================================
ALWAYS_INLINE double hsum_pd(__m256d v) {
    __m128d low  = _mm256_castpd256_pd128(v);
    __m128d high = _mm256_extractf128_pd(v, 1);
    low = _mm_add_pd(low, high);
    return _mm_cvtsd_f64(_mm_hadd_pd(low, low));
}

// ============================================================
// Makro generujace wyspecjalizowany kernel dla zadanych 
// ROW_BLOCK i COL_BLOCK. Kompilator zna rozmiary blokow
// w czasie kompilacji, co umozliwia:
// - optymalizacje petli (trip count znany)
// - lepsza alokacje rejestrow
// - eliminacje sprawdzen warunkow
// ============================================================

#define DEFINE_BLOCKED_KERNEL(NAME, ROW_BLK, COL_BLK, USE_PREFETCH)              \
ALWAYS_INLINE HOT void NAME(                                                      \
    int rows, int cols, double alpha,                                             \
    const double* __restrict A_ptr,                                               \
    const double* __restrict x_ptr,                                               \
    double* __restrict y_ptr)                                                     \
{                                                                                 \
    const double* A = ASSUME_ALIGNED(A_ptr, 32);                                  \
    const double* x = ASSUME_ALIGNED(x_ptr, 32);                                  \
    double* y = ASSUME_ALIGNED(y_ptr, 32);                                        \
                                                                                  \
    _Pragma("GCC unroll 2")                                                       \
    for (int jj = 0; jj < cols; jj += COL_BLK) {                                  \
        const int col_end = (jj + COL_BLK < cols) ? jj + COL_BLK : cols;          \
                                                                                  \
        if (USE_PREFETCH) {                                                       \
            for (int p = jj; p < col_end; p += 8) {                               \
                __builtin_prefetch(&x[p], 0, 3);                                  \
            }                                                                     \
        }                                                                         \
                                                                                  \
        for (int ii = 0; ii < rows; ii += ROW_BLK) {                              \
            const int row_end = (ii + ROW_BLK < rows) ? ii + ROW_BLK : rows;      \
                                                                                  \
            int i = ii;                                                           \
            _Pragma("GCC unroll 4")                                               \
            for (; i + 3 < row_end; i += 4) {                                     \
                __m256d sum0 = _mm256_setzero_pd();                               \
                __m256d sum1 = _mm256_setzero_pd();                               \
                __m256d sum2 = _mm256_setzero_pd();                               \
                __m256d sum3 = _mm256_setzero_pd();                               \
                                                                                  \
                if (USE_PREFETCH && i + 4 < row_end) {                            \
                    __builtin_prefetch(&A[(i + 4) * cols + jj], 0, 2);            \
                    __builtin_prefetch(&A[(i + 5) * cols + jj], 0, 2);            \
                }                                                                 \
                                                                                  \
                int j = jj;                                                       \
                _Pragma("GCC unroll 4")                                           \
                for (; j + 7 < col_end; j += 8) {                                 \
                    __m256d xv0 = _mm256_load_pd(&x[j]);                          \
                    __m256d xv1 = _mm256_load_pd(&x[j + 4]);                      \
                                                                                  \
                    __m256d a00 = _mm256_load_pd(&A[(i+0) * cols + j]);           \
                    __m256d a01 = _mm256_load_pd(&A[(i+0) * cols + j + 4]);       \
                    sum0 = _mm256_fmadd_pd(a00, xv0, sum0);                       \
                    sum0 = _mm256_fmadd_pd(a01, xv1, sum0);                       \
                                                                                  \
                    __m256d a10 = _mm256_load_pd(&A[(i+1) * cols + j]);           \
                    __m256d a11 = _mm256_load_pd(&A[(i+1) * cols + j + 4]);       \
                    sum1 = _mm256_fmadd_pd(a10, xv0, sum1);                       \
                    sum1 = _mm256_fmadd_pd(a11, xv1, sum1);                       \
                                                                                  \
                    __m256d a20 = _mm256_load_pd(&A[(i+2) * cols + j]);           \
                    __m256d a21 = _mm256_load_pd(&A[(i+2) * cols + j + 4]);       \
                    sum2 = _mm256_fmadd_pd(a20, xv0, sum2);                       \
                    sum2 = _mm256_fmadd_pd(a21, xv1, sum2);                       \
                                                                                  \
                    __m256d a30 = _mm256_load_pd(&A[(i+3) * cols + j]);           \
                    __m256d a31 = _mm256_load_pd(&A[(i+3) * cols + j + 4]);       \
                    sum3 = _mm256_fmadd_pd(a30, xv0, sum3);                       \
                    sum3 = _mm256_fmadd_pd(a31, xv1, sum3);                       \
                }                                                                 \
                                                                                  \
                for (; j + 3 < col_end; j += 4) {                                 \
                    __m256d xv = _mm256_load_pd(&x[j]);                           \
                    sum0 = _mm256_fmadd_pd(_mm256_load_pd(&A[(i+0)*cols+j]), xv, sum0); \
                    sum1 = _mm256_fmadd_pd(_mm256_load_pd(&A[(i+1)*cols+j]), xv, sum1); \
                    sum2 = _mm256_fmadd_pd(_mm256_load_pd(&A[(i+2)*cols+j]), xv, sum2); \
                    sum3 = _mm256_fmadd_pd(_mm256_load_pd(&A[(i+3)*cols+j]), xv, sum3); \
                }                                                                 \
                                                                                  \
                double t0 = 0, t1 = 0, t2 = 0, t3 = 0;                            \
                for (; j < col_end; j++) {                                        \
                    double xj = x[j];                                             \
                    t0 += A[(i+0) * cols + j] * xj;                               \
                    t1 += A[(i+1) * cols + j] * xj;                               \
                    t2 += A[(i+2) * cols + j] * xj;                               \
                    t3 += A[(i+3) * cols + j] * xj;                               \
                }                                                                 \
                                                                                  \
                y[i+0] += alpha * (hsum_pd(sum0) + t0);                           \
                y[i+1] += alpha * (hsum_pd(sum1) + t1);                           \
                y[i+2] += alpha * (hsum_pd(sum2) + t2);                           \
                y[i+3] += alpha * (hsum_pd(sum3) + t3);                           \
            }                                                                     \
                                                                                  \
            for (; i < row_end; i++) {                                            \
                __m256d sum = _mm256_setzero_pd();                                \
                int j = jj;                                                       \
                for (; j + 3 < col_end; j += 4) {                                 \
                    sum = _mm256_fmadd_pd(                                        \
                        _mm256_load_pd(&A[i * cols + j]),                         \
                        _mm256_load_pd(&x[j]),                                    \
                        sum                                                       \
                    );                                                            \
                }                                                                 \
                double tail = 0;                                                  \
                for (; j < col_end; j++) {                                        \
                    tail += A[i * cols + j] * x[j];                               \
                }                                                                 \
                y[i] += alpha * (hsum_pd(sum) + tail);                            \
            }                                                                     \
        }                                                                         \
    }                                                                             \
}

// ============================================================
// Kernel L1: male macierze, brak prefetch (dane juz w cache)
// ROW_BLOCK=64, COL_BLOCK=512 (ze strojenia)
// ============================================================
DEFINE_BLOCKED_KERNEL(gemv_kernel_l1_impl, L1_ROW_BLOCK, L1_COL_BLOCK, 0)

// ============================================================
// Kernel L2: srednie macierze, lekki prefetch
// ROW_BLOCK=128, COL_BLOCK=256 (ze strojenia)
// ============================================================
DEFINE_BLOCKED_KERNEL(gemv_kernel_l2_impl, L2_ROW_BLOCK, L2_COL_BLOCK, 1)

// ============================================================
// Kernel L3: duze macierze, agresywny prefetch
// ROW_BLOCK=64, COL_BLOCK=512 (ze strojenia)
// ============================================================
DEFINE_BLOCKED_KERNEL(gemv_kernel_l3_impl, L3_ROW_BLOCK, L3_COL_BLOCK, 1)

// ============================================================
// Wrappery dla kerneli - utrzymuja czytelna strukture kodu
// przy jednoczesnym zachowaniu mozliwosci profilowania
// ============================================================

HOT static void gemv_kernel_l1(int rows, int cols, double alpha,
                               const double* __restrict A,
                               const double* __restrict x,
                               double* __restrict y) {
    gemv_kernel_l1_impl(rows, cols, alpha, A, x, y);
}

HOT static void gemv_kernel_l2(int rows, int cols, double alpha,
                               const double* __restrict A,
                               const double* __restrict x,
                               double* __restrict y) {
    gemv_kernel_l2_impl(rows, cols, alpha, A, x, y);
}

HOT static void gemv_kernel_l3(int rows, int cols, double alpha,
                               const double* __restrict A,
                               const double* __restrict x,
                               double* __restrict y) {
    gemv_kernel_l3_impl(rows, cols, alpha, A, x, y);
}

// ============================================================
// Skalowanie wektora y przez beta - wyodrebnione dla czytelnosci
// ============================================================
ALWAYS_INLINE void scale_y(double* __restrict y, int rows, double beta) {
    if (beta == 0.0) {
        memset(y, 0, (size_t)rows * sizeof(double));
    } else if (beta != 1.0) {
        double* y_aligned = ASSUME_ALIGNED(y, 32);
        #pragma GCC unroll 8
        for (int i = 0; i < rows; i++) {
            y_aligned[i] *= beta;
        }
    }
}

// ============================================================
// Glowna funkcja GEMV z automatycznym wyborem kernela
// __restrict informuje kompilator ze wskazniki nie nachodza
// ============================================================
HOT void gemv_avx_fma_blocked(int rows, int cols, double alpha,
                              const double* __restrict A,
                              const double* __restrict x,
                              double beta,
                              double* __restrict y) {
    
    scale_y(y, rows, beta);
    
    // Oblicz working set w bajtach
    const size_t working_set = (size_t)rows * cols * sizeof(double) + 
                               (size_t)cols * sizeof(double) + 
                               (size_t)rows * sizeof(double);
    
    // Wybor kernela na podstawie rozmiaru danych
    // Stale enum sa znane w czasie kompilacji
    if (working_set <= L1_THRESHOLD) {
        gemv_kernel_l1(rows, cols, alpha, A, x, y);
    } else if (working_set <= L2_THRESHOLD) {
        gemv_kernel_l2(rows, cols, alpha, A, x, y);
    } else {
        gemv_kernel_l3(rows, cols, alpha, A, x, y);
    }
}
