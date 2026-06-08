#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "common.h"
#include "gemv.h"
#include "openblas/cblas.h"

extern void openblas_set_num_threads(int num_threads);
extern int  openblas_get_num_threads(void);
extern void goto_set_num_threads(int num_threads);

#define NUM_IMPLEMENTATIONS 9
#define NUM_RUNS 5
#define WARMUP_ITERATIONS 10

typedef void (*gemv_func)(int, int, double, const double*, const double*, double, double*);

typedef struct {
    const char* name;
    gemv_func   func;
    double      runs[NUM_RUNS];
    double      median;
    double      min;
    double      max;
    double      stddev;
    double      max_error;
} Impl;

typedef struct {
    const char* name;
    int rows;
    int cols;
    int iterations;
} BuiltinPreset;

static BuiltinPreset builtin_presets[] = {
    {"tiny",   64,   64,   5000000},
    {"small",  256,  256,  200000},
    {"medium", 1024, 1024, 10000},
    {"large",  4096, 4096, 300},
    {"wide",   256,  8192, 5000},
    {"tall",   8192, 256,  5000},
};
static const int num_builtin = sizeof(builtin_presets) / sizeof(builtin_presets[0]);

typedef void (*blis_dgemv_f77_func)(const char*, const int*, const int*,
                                    const double*, const double*, const int*,
                                    const double*, const int*,
                                    const double*, double*, const int*);
static blis_dgemv_f77_func blis_dgemv_f77 = NULL;

static void openblas_wrapper(int rows, int cols, double alpha,
                             const double* A, const double* x,
                             double beta, double* y) {
    cblas_dgemv(CblasRowMajor, CblasNoTrans, rows, cols, alpha,
                A, cols, x, 1, beta, y, 1);
}

static void blis_wrapper(int rows, int cols, double alpha,
                         const double* A, const double* x,
                         double beta, double* y) {
    if (!blis_dgemv_f77) return;
    char trans = 'T';
    int m = cols, n = rows, lda = cols, incx = 1, incy = 1;
    blis_dgemv_f77(&trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
}

#define BUDGET_PER_RUN_SEC 1.0
#define MAX_SECONDS_PER_CALL 20.0

static int measure_one(Impl* impl, int rows, int cols, int iterations,
                       const double* A, const double* x,
                       double* y, const double* y_ref,
                       double alpha, double beta) {
    double work = 2.0 * rows * cols;
    if (impl->median > 0.0) {
        double predicted = work / (impl->median * 1e9);
        if (predicted > MAX_SECONDS_PER_CALL) {
            impl->median = impl->min = impl->max = impl->stddev = 0.0;
            impl->max_error = 0.0;
            return -1;
        }
    }

    double t_cal = now_seconds();
    impl->func(rows, cols, alpha, A, x, beta, y);
    double t_one = now_seconds() - t_cal;

    if (t_one > MAX_SECONDS_PER_CALL) {
        impl->median = impl->min = impl->max = impl->stddev = 0.0;
        impl->max_error = max_abs_diff_d(y, y_ref, (size_t)rows);
        return 0;
    }

    int iters = iterations;
    int max_by_budget = (int)(BUDGET_PER_RUN_SEC / t_one);
    if (max_by_budget < 1) max_by_budget = 1;
    if (max_by_budget < iters) iters = max_by_budget;

    int runs = NUM_RUNS;
    if (t_one > 2.0) runs = 1;
    else if (t_one > 0.5) runs = 2;

    int warmup = (t_one < 0.05) ? WARMUP_ITERATIONS : 0;
    for (int w = 0; w < warmup; w++)
        impl->func(rows, cols, alpha, A, x, beta, y);

    double total_flops = 2.0 * rows * cols;
    double min_g = 1e30, max_g = 0.0;

    for (int run = 0; run < runs; run++) {
        double t0 = now_seconds();
        for (int i = 0; i < iters; i++)
            impl->func(rows, cols, alpha, A, x, beta, y);
        double dt = now_seconds() - t0;
        double gflops = (total_flops * iters) / (dt * 1e9);
        impl->runs[run] = gflops;
        if (gflops < min_g) min_g = gflops;
        if (gflops > max_g) max_g = gflops;
    }
    for (int r = runs; r < NUM_RUNS; r++) impl->runs[r] = impl->runs[0];

    impl->median = compute_median(impl->runs, runs);
    impl->min = min_g;
    impl->max = max_g;
    impl->stddev = (runs > 1) ? compute_stddev(impl->runs, runs, impl->median) : 0.0;
    impl->max_error = max_abs_diff_d(y, y_ref, (size_t)rows);
    return 1;
}

static void run_case(int rows, int cols, int iterations, Impl* impls) {
    double* A = (double*)aligned_alloc(32, (size_t)rows * cols * sizeof(double));
    double* x = (double*)aligned_alloc(32, cols * sizeof(double));
    double* y = (double*)aligned_alloc(32, rows * sizeof(double));
    double* y_ref = (double*)aligned_alloc(32, rows * sizeof(double));
    if (!A || !x || !y || !y_ref) { fprintf(stderr, "alloc failed\n"); exit(1); }

    srand(42);
    gen_double_arr(A, (size_t)rows * cols);
    gen_double_arr(x, cols);

    double alpha = 1.0, beta = 0.0;
    memset(y_ref, 0, rows * sizeof(double));
    openblas_wrapper(rows, cols, alpha, A, x, beta, y_ref);

    for (int i = 0; i < NUM_IMPLEMENTATIONS; i++) {
        printf("  [%d/%d] %-20s", i + 1, NUM_IMPLEMENTATIONS, impls[i].name);
        fflush(stdout);
        if (impls[i].func == blis_wrapper && !blis_dgemv_f77) {
            printf(" SKIPPED (AOCL-BLAS not loaded)\n");
            continue;
        }
        int ok = measure_one(&impls[i], rows, cols, iterations, A, x, y, y_ref, alpha, beta);
        if (ok == -1) {
            printf(" SKIPPED (predicted > %.0fs based on prior preset)\n", (double)MAX_SECONDS_PER_CALL);
        } else if (ok == 0) {
            printf(" SKIPPED (single call > %.0fs)\n", (double)MAX_SECONDS_PER_CALL);
        } else {
            printf(" med=%.2f [%.2f-%.2f] sd=%.2f\n",
                   impls[i].median, impls[i].min, impls[i].max, impls[i].stddev);
        }
    }

    free(A); free(x); free(y); free(y_ref);
}

static void print_table(const Impl* impls, int rows, int cols) {
    (void)rows; (void)cols;
    double naive_g = impls[0].median;
    double blas_g = impls[NUM_IMPLEMENTATIONS - 2].median;
    printf("\n-------------------------------------------------------------------------------------\n");
    printf("%-20s | %6s | %6s | %11s | %8s | %7s | %s\n",
           "Implementacja", "Median", "StdDev", "Min-Max", "vs Naive", "vs BLAS", "Blad");
    printf("---------------------|--------|--------|-------------|----------|---------|----------\n");
    for (int i = 0; i < NUM_IMPLEMENTATIONS; i++) {
        printf("%-20s | %6.2f | %6.2f | %5.2f-%-5.2f | %7.2fx | %6.2fx | %.1e\n",
               impls[i].name, impls[i].median, impls[i].stddev,
               impls[i].min, impls[i].max,
               impls[i].median / naive_g,
               impls[i].median / blas_g,
               impls[i].max_error);
    }
    printf("-------------------------------------------------------------------------------------\n");

    int best = 0;
    double best_g = 0.0;
    for (int i = 0; i < NUM_IMPLEMENTATIONS - 2; i++) {
        if (impls[i].median > best_g) { best_g = impls[i].median; best = i; }
    }
    double vs_blas = impls[best].median / blas_g;
    if (vs_blas >= 1.0)
        printf("\nNajlepsza: %s (%.2fx szybsza niz OpenBLAS)\n", impls[best].name, vs_blas);
    else
        printf("\nNajlepsza: %s (%.2fx wolniejsza niz OpenBLAS)\n", impls[best].name, 1.0 / vs_blas);
}

static void snapshot(PresetResult* out, const char* name, int rows, int cols,
                     const Impl* impls) {
    char params[64];
    snprintf(params, sizeof(params), "\"rows\": %d, \"cols\": %d", rows, cols);
    ImplResult tmp[NUM_IMPLEMENTATIONS];
    for (int i = 0; i < NUM_IMPLEMENTATIONS; i++) {
        memset(&tmp[i], 0, sizeof(tmp[i]));
        strncpy(tmp[i].name, impls[i].name, sizeof(tmp[i].name) - 1);
        tmp[i].median = impls[i].median;
        tmp[i].min = impls[i].min;
        tmp[i].max = impls[i].max;
        tmp[i].stddev = impls[i].stddev;
        tmp[i].max_error = impls[i].max_error;
    }
    bench_snapshot(out, name, params, tmp, NUM_IMPLEMENTATIONS);
}

static void run_preset(const char* name, int rows, int cols, int iterations,
                       Impl* impls, PresetResult* out) {
    printf("\n[ Preset: %s | %dx%d | %d iteracji ]\n\n", name, rows, cols, iterations);
    printf("Uruchamianie benchmarkow...\n");
    run_case(rows, cols, iterations, impls);
    print_table(impls, rows, cols);
    if (out) snapshot(out, name, rows, cols, impls);
}

static void print_usage(const char* prog) {
    printf("Uzycie:\n");
    printf("  %s                          - uruchamia preset 'medium'\n", prog);
    printf("  %s <preset>                 - uruchamia wbudowany preset\n", prog);
    printf("  %s all                      - uruchamia wszystkie wbudowane presety\n", prog);
    printf("  %s --preset-file <path>     - presety z pliku INI (kernel = gemv)\n", prog);
    printf("  %s --custom <iter> <r> <c>  - parametry recznie\n", prog);
    printf("  %s [--json <path>] ...      - zapisz wyniki do JSON\n", prog);
    printf("\nWbudowane presety:\n");
    for (int i = 0; i < num_builtin; i++) {
        printf("  %-8s - %dx%d, %d iteracji\n",
               builtin_presets[i].name, builtin_presets[i].rows,
               builtin_presets[i].cols, builtin_presets[i].iterations);
    }
}

int main(int argc, char* argv[]) {
    int nthreads = omp_get_max_threads();
    openblas_set_num_threads(nthreads);
    goto_set_num_threads(nthreads);
    int has_blis = blis_loader_init(nthreads);
    if (has_blis) blis_dgemv_f77 = (blis_dgemv_f77_func)blis_loader_sym("dgemv_");

    Impl impls[NUM_IMPLEMENTATIONS] = {
        {"Naive",           gemv_naive,           {0}, 0, 0, 0, 0, 0},
        {"SIMD",            gemv_simd,            {0}, 0, 0, 0, 0, 0},
        {"SIMD + Prefetch", gemv_simd_prefetch,   {0}, 0, 0, 0, 0, 0},
        {"AVX+FMA Blocked", gemv_avx_fma_blocked, {0}, 0, 0, 0, 0, 0},
        {"AVX+FMA V2",      gemv_avx_fma_v2,      {0}, 0, 0, 0, 0, 0},
        {"AVX+FMA V3",      gemv_avx_fma_v3,      {0}, 0, 0, 0, 0, 0},
        {"AVX+FMA V3_OMP",  gemv_avx_fma_v3_omp,  {0}, 0, 0, 0, 0, 0},
        {"OpenBLAS",        openblas_wrapper,     {0}, 0, 0, 0, 0, 0},
        {"AOCL-BLAS",            blis_wrapper,         {0}, 0, 0, 0, 0, 0},
    };
    char openblas_name[32], blis_name_buf[32];
    snprintf(openblas_name, sizeof(openblas_name), "OpenBLAS (%dT)", nthreads);
    snprintf(blis_name_buf, sizeof(blis_name_buf), "AOCL-BLAS (%dT)%s",
             nthreads, blis_dgemv_f77 ? "" : " N/A");
    impls[7].name = openblas_name;
    impls[8].name = blis_name_buf;

    const char* json_path = NULL;
    const char* preset_file = NULL;
    int custom_iter = 0, custom_rows = 0, custom_cols = 0;
    int run_all = 0;
    const char* single_preset = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            json_path = argv[++i];
        } else if (strcmp(argv[i], "--preset-file") == 0 && i + 1 < argc) {
            preset_file = argv[++i];
        } else if (strcmp(argv[i], "--custom") == 0 && i + 3 < argc) {
            custom_iter = atoi(argv[++i]);
            custom_rows = atoi(argv[++i]);
            custom_cols = atoi(argv[++i]);
        } else if (strcmp(argv[i], "all") == 0) {
            run_all = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            blis_loader_shutdown();
            return 0;
        } else {
            single_preset = argv[i];
        }
    }

    print_system_header("GEMV Benchmark");
    printf("Threads: OMP=%d, OpenBLAS=%d, AOCL-BLAS=%s\n",
           omp_get_max_threads(), openblas_get_num_threads(),
           has_blis ? "loaded" : "not found");

    PresetResult* results = NULL;
    int num_results = 0;
    int rc = 0;

    if (preset_file) {
        Preset* head = preset_load(preset_file);
        if (!head) { rc = 1; goto cleanup; }
        int total = preset_count(head);
        results = (PresetResult*)calloc(total, sizeof(PresetResult));
        for (Preset* p = head; p; p = p->next) {
            const char* k = preset_get(p, "kernel");
            if (!k || strcmp(k, "gemv") != 0) continue;
            int rows = preset_get_int(p, "rows", 0);
            int cols = preset_get_int(p, "cols", 0);
            int iter = preset_get_int(p, "iterations", 0);
            if (rows <= 0 || cols <= 0 || iter <= 0) {
                fprintf(stderr, "preset '%s': rows/cols/iterations missing or invalid\n", p->name);
                continue;
            }
            run_preset(p->name, rows, cols, iter, impls, &results[num_results++]);
        }
        preset_free(head);
    } else if (custom_iter > 0) {
        results = (PresetResult*)calloc(1, sizeof(PresetResult));
        run_preset("custom", custom_rows, custom_cols, custom_iter, impls, &results[0]);
        num_results = 1;
    } else if (run_all) {
        results = (PresetResult*)calloc(num_builtin, sizeof(PresetResult));
        for (int i = 0; i < num_builtin; i++) {
            BuiltinPreset* bp = &builtin_presets[i];
            run_preset(bp->name, bp->rows, bp->cols, bp->iterations, impls, &results[i]);
        }
        num_results = num_builtin;
    } else {
        const char* name = single_preset ? single_preset : "medium";
        int found = -1;
        for (int i = 0; i < num_builtin; i++)
            if (strcmp(name, builtin_presets[i].name) == 0) { found = i; break; }
        if (found < 0) {
            fprintf(stderr, "Nieznany preset: %s\n", name);
            print_usage(argv[0]);
            rc = 1; goto cleanup;
        }
        results = (PresetResult*)calloc(1, sizeof(PresetResult));
        BuiltinPreset* bp = &builtin_presets[found];
        run_preset(bp->name, bp->rows, bp->cols, bp->iterations, impls, &results[0]);
        num_results = 1;
    }

    bench_print_summary(results, num_results, NUM_IMPLEMENTATIONS, NUM_RUNS);

    if (json_path && num_results > 0) {
        json_write_results(json_path, "gemv", nthreads, NUM_RUNS, results, num_results);
        printf("\nResults written to %s\n", json_path);
    }

cleanup:
    if (results) { bench_free_results(results, num_results); free(results); }
    blis_loader_shutdown();
    return rc;
}
