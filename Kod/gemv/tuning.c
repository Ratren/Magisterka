#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>
#include <xmmintrin.h>

#define WARMUP_ITERS 10
#define TIMED_ITERS 10000

// ============================================================
// Struktury danych
// ============================================================

typedef struct {
    int row_block;
    int col_block;
    int unroll_rows;
    int prefetch_dist;
} TuningParams;

typedef struct {
    int rows;
    int cols;
    const char* name;
    const char* cache_level;
} MatrixSize;

typedef struct {
    TuningParams params;
    double gflops;
} TuningResult;

// ============================================================
// Redukcja horyzontalna
// ============================================================
static inline double hsum_pd(__m256d v) {
    __m128d low  = _mm256_castpd256_pd128(v);
    __m128d high = _mm256_extractf128_pd(v, 1);
    low = _mm_add_pd(low, high);
    return _mm_cvtsd_f64(_mm_hadd_pd(low, low));
}

// ============================================================
// Parametryzowany kernel do tuningu
// ============================================================
static void gemv_tuned(int rows, int cols, double alpha,
                       const double* A, const double* x,
                       double* y, const TuningParams* p) {
    
    memset(y, 0, rows * sizeof(double));
    
    for (int jj = 0; jj < cols; jj += p->col_block) {
        int col_end = (jj + p->col_block < cols) ? jj + p->col_block : cols;
        
        // Prefetch x
        if (p->prefetch_dist > 0) {
            for (int pp = jj; pp < col_end; pp += 8) {
                _mm_prefetch((const char*)&x[pp], _MM_HINT_T0);
            }
        }
        
        for (int ii = 0; ii < rows; ii += p->row_block) {
            int row_end = (ii + p->row_block < rows) ? ii + p->row_block : rows;
            
            if (p->unroll_rows >= 4) {
                // Unroll 4 wiersze
                int i = ii;
                for (; i + 3 < row_end; i += 4) {
                    __m256d sum0 = _mm256_setzero_pd();
                    __m256d sum1 = _mm256_setzero_pd();
                    __m256d sum2 = _mm256_setzero_pd();
                    __m256d sum3 = _mm256_setzero_pd();
                    
                    int j = jj;
                    for (; j + 7 < col_end; j += 8) {
                        if (p->prefetch_dist > 0) {
                            _mm_prefetch((const char*)&A[(i+0) * cols + j + p->prefetch_dist], _MM_HINT_T0);
                        }
                        
                        __m256d xv0 = _mm256_load_pd(&x[j]);
                        __m256d xv1 = _mm256_load_pd(&x[j + 4]);
                        
                        __m256d a00 = _mm256_load_pd(&A[(i+0) * cols + j]);
                        __m256d a01 = _mm256_load_pd(&A[(i+0) * cols + j + 4]);
                        sum0 = _mm256_fmadd_pd(a00, xv0, sum0);
                        sum0 = _mm256_fmadd_pd(a01, xv1, sum0);
                        
                        __m256d a10 = _mm256_load_pd(&A[(i+1) * cols + j]);
                        __m256d a11 = _mm256_load_pd(&A[(i+1) * cols + j + 4]);
                        sum1 = _mm256_fmadd_pd(a10, xv0, sum1);
                        sum1 = _mm256_fmadd_pd(a11, xv1, sum1);
                        
                        __m256d a20 = _mm256_load_pd(&A[(i+2) * cols + j]);
                        __m256d a21 = _mm256_load_pd(&A[(i+2) * cols + j + 4]);
                        sum2 = _mm256_fmadd_pd(a20, xv0, sum2);
                        sum2 = _mm256_fmadd_pd(a21, xv1, sum2);
                        
                        __m256d a30 = _mm256_load_pd(&A[(i+3) * cols + j]);
                        __m256d a31 = _mm256_load_pd(&A[(i+3) * cols + j + 4]);
                        sum3 = _mm256_fmadd_pd(a30, xv0, sum3);
                        sum3 = _mm256_fmadd_pd(a31, xv1, sum3);
                    }
                    
                    for (; j + 3 < col_end; j += 4) {
                        __m256d xv = _mm256_load_pd(&x[j]);
                        sum0 = _mm256_fmadd_pd(_mm256_load_pd(&A[(i+0) * cols + j]), xv, sum0);
                        sum1 = _mm256_fmadd_pd(_mm256_load_pd(&A[(i+1) * cols + j]), xv, sum1);
                        sum2 = _mm256_fmadd_pd(_mm256_load_pd(&A[(i+2) * cols + j]), xv, sum2);
                        sum3 = _mm256_fmadd_pd(_mm256_load_pd(&A[(i+3) * cols + j]), xv, sum3);
                    }
                    
                    double t0 = 0, t1 = 0, t2 = 0, t3 = 0;
                    for (; j < col_end; j++) {
                        double xj = x[j];
                        t0 += A[(i+0) * cols + j] * xj;
                        t1 += A[(i+1) * cols + j] * xj;
                        t2 += A[(i+2) * cols + j] * xj;
                        t3 += A[(i+3) * cols + j] * xj;
                    }
                    
                    y[i+0] += alpha * (hsum_pd(sum0) + t0);
                    y[i+1] += alpha * (hsum_pd(sum1) + t1);
                    y[i+2] += alpha * (hsum_pd(sum2) + t2);
                    y[i+3] += alpha * (hsum_pd(sum3) + t3);
                }
                
                // Pozostale wiersze
                for (; i < row_end; i++) {
                    __m256d sum = _mm256_setzero_pd();
                    int j = jj;
                    for (; j + 3 < col_end; j += 4) {
                        sum = _mm256_fmadd_pd(
                            _mm256_load_pd(&A[i * cols + j]),
                            _mm256_load_pd(&x[j]),
                            sum
                        );
                    }
                    double tail = 0;
                    for (; j < col_end; j++) {
                        tail += A[i * cols + j] * x[j];
                    }
                    y[i] += alpha * (hsum_pd(sum) + tail);
                }
            } else if (p->unroll_rows >= 2) {
                // Unroll 2 wiersze
                int i = ii;
                for (; i + 1 < row_end; i += 2) {
                    __m256d sum0 = _mm256_setzero_pd();
                    __m256d sum1 = _mm256_setzero_pd();
                    
                    int j = jj;
                    for (; j + 3 < col_end; j += 4) {
                        __m256d xv = _mm256_load_pd(&x[j]);
                        sum0 = _mm256_fmadd_pd(_mm256_load_pd(&A[(i+0) * cols + j]), xv, sum0);
                        sum1 = _mm256_fmadd_pd(_mm256_load_pd(&A[(i+1) * cols + j]), xv, sum1);
                    }
                    
                    double t0 = 0, t1 = 0;
                    for (; j < col_end; j++) {
                        double xj = x[j];
                        t0 += A[(i+0) * cols + j] * xj;
                        t1 += A[(i+1) * cols + j] * xj;
                    }
                    
                    y[i+0] += alpha * (hsum_pd(sum0) + t0);
                    y[i+1] += alpha * (hsum_pd(sum1) + t1);
                }
                
                for (; i < row_end; i++) {
                    __m256d sum = _mm256_setzero_pd();
                    int j = jj;
                    for (; j + 3 < col_end; j += 4) {
                        sum = _mm256_fmadd_pd(
                            _mm256_load_pd(&A[i * cols + j]),
                            _mm256_load_pd(&x[j]),
                            sum
                        );
                    }
                    double tail = 0;
                    for (; j < col_end; j++) {
                        tail += A[i * cols + j] * x[j];
                    }
                    y[i] += alpha * (hsum_pd(sum) + tail);
                }
            } else {
                // Bez unrollingu
                for (int i = ii; i < row_end; i++) {
                    __m256d sum = _mm256_setzero_pd();
                    int j = jj;
                    for (; j + 3 < col_end; j += 4) {
                        sum = _mm256_fmadd_pd(
                            _mm256_load_pd(&A[i * cols + j]),
                            _mm256_load_pd(&x[j]),
                            sum
                        );
                    }
                    double tail = 0;
                    for (; j < col_end; j++) {
                        tail += A[i * cols + j] * x[j];
                    }
                    y[i] += alpha * (hsum_pd(sum) + tail);
                }
            }
        }
    }
}

// ============================================================
// OpenBLAS wrapper dla porownania
// ============================================================
extern void cblas_dgemv(int order, int trans, int m, int n,
                        double alpha, const double* a, int lda,
                        const double* x, int incx,
                        double beta, double* y, int incy);

static void gemv_openblas(int rows, int cols, double alpha,
                          const double* A, const double* x,
                          double* y) {
    memset(y, 0, rows * sizeof(double));
    cblas_dgemv(101, 111, rows, cols, alpha, A, cols, x, 1, 0.0, y, 1);
}

// ============================================================
// Generowanie danych
// ============================================================
static void gen_double_arr(double* arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
}

// ============================================================
// Benchmark
// ============================================================
static double benchmark_config(int rows, int cols,
                               double* A, double* x, double* y,
                               const TuningParams* p) {
    double alpha = 1.0;
    
    // Warmup
    for (int w = 0; w < WARMUP_ITERS; w++) {
        gemv_tuned(rows, cols, alpha, A, x, y, p);
    }
    
    // Timed
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < TIMED_ITERS; i++) {
        gemv_tuned(rows, cols, alpha, A, x, y, p);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double elapsed = (end.tv_sec - start.tv_sec) + 
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    
    long total_flops = 2L * rows * cols * TIMED_ITERS;
    return (double)total_flops / (elapsed * 1e9);
}

static double benchmark_openblas(int rows, int cols,
                                 double* A, double* x, double* y) {
    double alpha = 1.0;
    
    for (int w = 0; w < WARMUP_ITERS; w++) {
        gemv_openblas(rows, cols, alpha, A, x, y);
    }
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < TIMED_ITERS; i++) {
        gemv_openblas(rows, cols, alpha, A, x, y);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double elapsed = (end.tv_sec - start.tv_sec) + 
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    
    long total_flops = 2L * rows * cols * TIMED_ITERS;
    return (double)total_flops / (elapsed * 1e9);
}

// ============================================================
// Main
// ============================================================
int main() {
    printf("GEMV AVX+FMA Tuning - starting...\n\n");
    
    MatrixSize sizes[] = {
        {64,   64,   "64x64",     "L1"},
        {256,  256,  "256x256",   "L2"},
        {1024, 1024, "1024x1024", "L3"},
        {4096, 4096, "4096x4096", "RAM"},
    };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    int row_blocks[] = {4, 8, 16, 32, 64, 128};
    int col_blocks[] = {32, 64, 128, 256, 512};
    int unroll_rows[] = {1, 2, 4};
    int prefetch_dists[] = {0, 32, 64, 128};
    
    int num_row_blocks = sizeof(row_blocks) / sizeof(row_blocks[0]);
    int num_col_blocks = sizeof(col_blocks) / sizeof(col_blocks[0]);
    int num_unrolls = sizeof(unroll_rows) / sizeof(unroll_rows[0]);
    int num_prefetch = sizeof(prefetch_dists) / sizeof(prefetch_dists[0]);
    
    int total_configs = num_row_blocks * num_col_blocks * num_unrolls * num_prefetch;
    printf("Testing %d configurations across %d matrix sizes...\n\n", 
           total_configs, num_sizes);
    
    FILE* f = fopen("tuning_results.txt", "w");
    if (!f) {
        fprintf(stderr, "Cannot open tuning_results.txt for writing\n");
        return 1;
    }
    
    // Naglowek
    fprintf(f, "================================================================\n");
    fprintf(f, "GEMV AVX+FMA Tuning Results - AMD Ryzen 5 5600 (Zen 3)\n");
    fprintf(f, "================================================================\n\n");
    
    fprintf(f, "THEORETICAL OPTIMAL VALUES (75%% cache utilization):\n");
    fprintf(f, "  L1 (32KB):  Working set ~24KB -> ROW_BLOCK=8,   COL_BLOCK=48-64\n");
    fprintf(f, "  L2 (512KB): Working set ~384KB -> ROW_BLOCK=64,  COL_BLOCK=192-256\n");
    fprintf(f, "  L3 (4MB):   Working set ~3MB   -> ROW_BLOCK=128, COL_BLOCK=512-1024\n\n");
    
    fprintf(f, "================================================================\n");
    fprintf(f, "                    EXPERIMENTAL RESULTS\n");
    fprintf(f, "================================================================\n\n");
    
    // Przechowuj najlepsze wyniki dla kazdego rozmiaru
    TuningResult best[4];
    double openblas_gflops[4];
    for (int s = 0; s < num_sizes; s++) {
        best[s].gflops = 0;
    }
    
    srand(42);
    
    // Alokuj pamiec dla najwiekszej macierzy
    int max_rows = 4096, max_cols = 4096;
    double* A = (double*)aligned_alloc(32, max_rows * max_cols * sizeof(double));
    double* x = (double*)aligned_alloc(32, max_cols * sizeof(double));
    double* y = (double*)aligned_alloc(32, max_rows * sizeof(double));
    
    if (!A || !x || !y) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    // Dla kazdego rozmiaru macierzy
    for (int s = 0; s < num_sizes; s++) {
        int rows = sizes[s].rows;
        int cols = sizes[s].cols;
        
        printf("Testing %s (%s)...\n", sizes[s].name, sizes[s].cache_level);
        fprintf(f, "Matrix: %s (fits in %s cache)\n", sizes[s].name, sizes[s].cache_level);
        fprintf(f, "----------------------------------------------------------------\n");
        fprintf(f, "ROW_BLK | COL_BLK | UNROLL | PREFETCH | GFLOPS\n");
        fprintf(f, "--------|---------|--------|----------|--------\n");
        
        // Generuj dane
        gen_double_arr(A, rows * cols);
        gen_double_arr(x, cols);
        
        // OpenBLAS baseline
        openblas_gflops[s] = benchmark_openblas(rows, cols, A, x, y);
        
        int config_count = 0;
        
        // Grid search
        for (int ri = 0; ri < num_row_blocks; ri++) {
            for (int ci = 0; ci < num_col_blocks; ci++) {
                for (int ui = 0; ui < num_unrolls; ui++) {
                    for (int pi = 0; pi < num_prefetch; pi++) {
                        TuningParams p = {
                            .row_block = row_blocks[ri],
                            .col_block = col_blocks[ci],
                            .unroll_rows = unroll_rows[ui],
                            .prefetch_dist = prefetch_dists[pi]
                        };
                        
                        double gflops = benchmark_config(rows, cols, A, x, y, &p);
                        
                        fprintf(f, "%7d | %7d | %6d | %8d | %6.2f\n",
                                p.row_block, p.col_block, p.unroll_rows,
                                p.prefetch_dist, gflops);
                        
                        if (gflops > best[s].gflops) {
                            best[s].gflops = gflops;
                            best[s].params = p;
                        }
                        
                        config_count++;
                        if (config_count % 30 == 0) {
                            printf("  Progress: %d/%d configs\n", config_count, total_configs);
                        }
                    }
                }
            }
        }
        
        fprintf(f, "----------------------------------------------------------------\n");
        fprintf(f, "OpenBLAS baseline: %.2f GFLOPS\n", openblas_gflops[s]);
        fprintf(f, "Best config: ROW=%d, COL=%d, UNROLL=%d, PREFETCH=%d -> %.2f GFLOPS\n",
                best[s].params.row_block, best[s].params.col_block,
                best[s].params.unroll_rows, best[s].params.prefetch_dist,
                best[s].gflops);
        fprintf(f, "Speedup vs OpenBLAS: %.2fx\n\n", best[s].gflops / openblas_gflops[s]);
        
        printf("  Done! Best: %.2f GFLOPS (OpenBLAS: %.2f)\n\n", 
               best[s].gflops, openblas_gflops[s]);
    }
    
    // Podsumowanie
    fprintf(f, "================================================================\n");
    fprintf(f, "                         SUMMARY\n");
    fprintf(f, "================================================================\n\n");
    
    fprintf(f, "BEST CONFIGURATIONS FOR EACH MATRIX SIZE:\n\n");
    
    for (int s = 0; s < num_sizes; s++) {
        fprintf(f, "%s (%s-bound):\n", sizes[s].name, sizes[s].cache_level);
        fprintf(f, "  Best params:  ROW_BLOCK=%d, COL_BLOCK=%d, UNROLL=%d, PREFETCH=%d\n",
                best[s].params.row_block, best[s].params.col_block,
                best[s].params.unroll_rows, best[s].params.prefetch_dist);
        fprintf(f, "  Performance:  %.2f GFLOPS\n", best[s].gflops);
        fprintf(f, "  OpenBLAS:     %.2f GFLOPS\n", openblas_gflops[s]);
        fprintf(f, "  Speedup:      %.2fx %s\n\n",
                best[s].gflops / openblas_gflops[s],
                best[s].gflops >= openblas_gflops[s] ? "(FASTER)" : "(slower)");
    }
    
    fprintf(f, "================================================================\n");
    fprintf(f, "           RECOMMENDED VALUES FOR gemv_avx_fma_blocked.c\n");
    fprintf(f, "================================================================\n\n");
    
    // L1 - uzyj best[0] (64x64)
    fprintf(f, "// L1 kernel (for matrices fitting in L1 cache)\n");
    fprintf(f, "#define L1_ROW_BLOCK    %d\n", best[0].params.row_block);
    fprintf(f, "#define L1_COL_BLOCK    %d\n\n", best[0].params.col_block);
    
    // L2 - uzyj best[1] (256x256)
    fprintf(f, "// L2 kernel (for matrices fitting in L2 cache)\n");
    fprintf(f, "#define L2_ROW_BLOCK    %d\n", best[1].params.row_block);
    fprintf(f, "#define L2_COL_BLOCK    %d\n\n", best[1].params.col_block);
    
    // L3 - uzyj sredniej z best[2] i best[3]
    fprintf(f, "// L3 kernel (for large matrices)\n");
    fprintf(f, "#define L3_ROW_BLOCK    %d\n", best[2].params.row_block);
    fprintf(f, "#define L3_COL_BLOCK    %d\n\n", best[2].params.col_block);
    
    fprintf(f, "// Prefetch distances\n");
    fprintf(f, "#define PREFETCH_L1_DIST    %d\n", best[0].params.prefetch_dist);
    fprintf(f, "#define PREFETCH_L2_DIST    %d\n", best[2].params.prefetch_dist);
    
    fprintf(f, "\n================================================================\n");
    
    fclose(f);
    
    free(A);
    free(x);
    free(y);
    
    printf("Tuning complete! Results saved to tuning_results.txt\n");
    
    return 0;
}
