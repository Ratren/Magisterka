#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "common.h"
#include "gemm.h"
#include "openblas/cblas.h"

extern void openblas_set_num_threads(int num_threads);
extern int  openblas_get_num_threads(void);
extern void goto_set_num_threads(int num_threads);

#define NUM_IMPLEMENTATIONS 8
#define NUM_RUNS 5
#define WARMUP_ITERATIONS 3

typedef void (*gemm_func)(int, int, int, double,
                          const double*, const double*,
                          double, double*);

typedef struct {
    const char* name;
    gemm_func   func;
    double      runs[NUM_RUNS];
    double      median;
    double      min;
    double      max;
    double      stddev;
    double      max_error;
} Impl;

typedef struct {
    const char* name;
    int M, N, K;
    int iterations;
} BuiltinPreset;

static BuiltinPreset builtin_presets[] = {
    {"tiny",   64,   64,   64,   1000},
    {"small",  256,  256,  256,  100},
    {"medium", 1024, 1024, 1024, 10},
    {"large",  4096, 4096, 4096, 2},
    {"rank_k", 2048, 2048, 128,  20},
    {"tall_K", 128,  2048, 2048, 20},
};
static const int num_builtin = sizeof(builtin_presets) / sizeof(builtin_presets[0]);

typedef void (*blis_dgemm_f77_func)(const char*, const char*,
                                    const int*, const int*, const int*,
                                    const double*, const double*, const int*,
                                    const double*, const int*,
                                    const double*, double*, const int*);
static blis_dgemm_f77_func blis_dgemm_f77 = NULL;

static void openblas_wrapper(int M, int N, int K, double alpha,
                             const double* A, const double* B,
                             double beta, double* C) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A, K, B, N, beta, C, N);
}

static void blis_wrapper(int M, int N, int K, double alpha,
                         const double* A, const double* B,
                         double beta, double* C) {
    if (!blis_dgemm_f77) return;
    char trans = 'N';
    int m = N, n = M, k = K, lda = N, ldb = K, ldc = N;
    blis_dgemm_f77(&trans, &trans, &m, &n, &k, &alpha,
                   B, &lda, A, &ldb, &beta, C, &ldc);
}

#define BUDGET_PER_RUN_SEC 1.0
#define MAX_SECONDS_PER_CALL 20.0

static int measure_one(Impl* impl, int M, int N, int K, int iterations,
                       const double* A, const double* B,
                       double* C, const double* C_ref,
                       double alpha, double beta) {
    double work = 2.0 * (double)M * (double)N * (double)K;
    if (impl->median > 0.0) {
        double predicted = work / (impl->median * 1e9);
        if (predicted > MAX_SECONDS_PER_CALL) {
            impl->median = impl->min = impl->max = impl->stddev = 0.0;
            impl->max_error = 0.0;
            return -1;
        }
    }

    double t_cal = now_seconds();
    impl->func(M, N, K, alpha, A, B, beta, C);
    double t_one = now_seconds() - t_cal;

    if (t_one > MAX_SECONDS_PER_CALL) {
        impl->median = 0.0;
        impl->min = impl->max = impl->stddev = 0.0;
        impl->max_error = max_abs_diff_d(C, C_ref, (size_t)M * N);
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
        impl->func(M, N, K, alpha, A, B, beta, C);

    double total_flops = 2.0 * (double)M * (double)N * (double)K;
    double min_g = 1e30, max_g = 0.0;

    for (int run = 0; run < runs; run++) {
        double t0 = now_seconds();
        for (int i = 0; i < iters; i++)
            impl->func(M, N, K, alpha, A, B, beta, C);
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
    impl->max_error = max_abs_diff_d(C, C_ref, (size_t)M * N);
    return 1;
}

static void run_case(int M, int N, int K, int iterations, Impl* impls) {
    size_t aSize = (size_t)M * K;
    size_t bSize = (size_t)K * N;
    size_t cSize = (size_t)M * N;

    double* A     = (double*)aligned_alloc(64, aSize * sizeof(double));
    double* B     = (double*)aligned_alloc(64, bSize * sizeof(double));
    double* C     = (double*)aligned_alloc(64, cSize * sizeof(double));
    double* C_ref = (double*)aligned_alloc(64, cSize * sizeof(double));
    if (!A || !B || !C || !C_ref) { fprintf(stderr, "alloc failed\n"); exit(1); }

    srand(42);
    gen_double_arr(A, aSize);
    gen_double_arr(B, bSize);

    double alpha = 1.0, beta = 0.0;
    memset(C_ref, 0, cSize * sizeof(double));
    openblas_wrapper(M, N, K, alpha, A, B, beta, C_ref);

    for (int i = 0; i < NUM_IMPLEMENTATIONS; i++) {
        printf("  [%d/%d] %-20s", i + 1, NUM_IMPLEMENTATIONS, impls[i].name);
        fflush(stdout);
        if (impls[i].func == blis_wrapper && !blis_dgemm_f77) {
            printf(" SKIPPED (BLIS not loaded)\n");
            continue;
        }
        memset(C, 0, cSize * sizeof(double));
        int ok = measure_one(&impls[i], M, N, K, iterations, A, B, C, C_ref, alpha, beta);
        if (ok == -1) {
            printf(" SKIPPED (predicted > %.0fs based on prior preset)\n", (double)MAX_SECONDS_PER_CALL);
        } else if (ok == 0) {
            printf(" SKIPPED (single call > %.0fs)\n", (double)MAX_SECONDS_PER_CALL);
        } else {
            printf(" med=%.2f [%.2f-%.2f] sd=%.2f err=%.1e\n",
                   impls[i].median, impls[i].min, impls[i].max, impls[i].stddev, impls[i].max_error);
        }
    }

    free(A); free(B); free(C); free(C_ref);
}

static void print_table(const Impl* impls) {
    double naive_g = impls[0].median;
    double blas_g  = impls[NUM_IMPLEMENTATIONS - 2].median;
    printf("\n-------------------------------------------------------------------------------------\n");
    printf("%-20s | %6s | %6s | %11s | %8s | %7s | %s\n",
           "Implementacja", "Median", "StdDev", "Min-Max", "vs Naive", "vs BLAS", "Blad");
    printf("---------------------|--------|--------|-------------|----------|---------|----------\n");
    for (int i = 0; i < NUM_IMPLEMENTATIONS; i++) {
        printf("%-20s | %6.2f | %6.2f | %5.2f-%-5.2f | %7.2fx | %6.2fx | %.1e\n",
               impls[i].name, impls[i].median, impls[i].stddev,
               impls[i].min, impls[i].max,
               naive_g > 0 ? impls[i].median / naive_g : 0.0,
               blas_g  > 0 ? impls[i].median / blas_g  : 0.0,
               impls[i].max_error);
    }
    printf("-------------------------------------------------------------------------------------\n");

    int best = 0;
    double best_g = 0.0;
    for (int i = 0; i < NUM_IMPLEMENTATIONS - 2; i++) {
        if (impls[i].median > best_g) { best_g = impls[i].median; best = i; }
    }
    if (blas_g > 0) {
        double vs_blas = impls[best].median / blas_g;
        if (vs_blas >= 1.0)
            printf("\nNajlepsza: %s (%.2fx szybsza niz OpenBLAS)\n", impls[best].name, vs_blas);
        else
            printf("\nNajlepsza: %s (%.2fx wolniejsza niz OpenBLAS)\n", impls[best].name, 1.0 / vs_blas);
    }
}

static void snapshot(PresetResult* out, const char* name, int M, int N, int K,
                     const Impl* impls) {
    char params[64];
    snprintf(params, sizeof(params), "\"m\": %d, \"n\": %d, \"k\": %d", M, N, K);
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

static void run_preset(const char* name, int M, int N, int K, int iterations,
                       Impl* impls, PresetResult* out) {
    printf("\n[ Preset: %s | %dx%dx%d | %d iteracji ]\n\n", name, M, N, K, iterations);
    printf("Uruchamianie benchmarkow...\n");
    run_case(M, N, K, iterations, impls);
    print_table(impls);
    if (out) snapshot(out, name, M, N, K, impls);
}

static void print_usage(const char* prog) {
    printf("Uzycie:\n");
    printf("  %s                              - uruchamia preset 'small'\n", prog);
    printf("  %s <preset>                     - uruchamia wbudowany preset\n", prog);
    printf("  %s all                          - uruchamia wszystkie wbudowane presety\n", prog);
    printf("  %s --preset-file <path>         - presety z pliku INI (kernel = gemm)\n", prog);
    printf("  %s --custom <iter> <M> <N> <K>  - parametry recznie\n", prog);
    printf("  %s [--json <path>] ...          - zapisz wyniki do JSON\n", prog);
    printf("\nWbudowane presety:\n");
    for (int i = 0; i < num_builtin; i++) {
        printf("  %-8s - %dx%dx%d, %d iteracji\n",
               builtin_presets[i].name, builtin_presets[i].M,
               builtin_presets[i].N, builtin_presets[i].K,
               builtin_presets[i].iterations);
    }
}

int main(int argc, char* argv[]) {
    int nthreads = omp_get_max_threads();
    openblas_set_num_threads(nthreads);
    goto_set_num_threads(nthreads);
    int has_blis = blis_loader_init(nthreads);
    if (has_blis) blis_dgemm_f77 = (blis_dgemm_f77_func)blis_loader_sym("dgemm_");

    Impl impls[NUM_IMPLEMENTATIONS] = {
        {"Naive",            gemm_naive,        {0}, 0, 0, 0, 0, 0},
        {"Loop Reorder ikj", gemm_ikj,          {0}, 0, 0, 0, 0, 0},
        {"Blocked",          gemm_blocked,      {0}, 0, 0, 0, 0, 0},
        {"Microkernel 6x8",  gemm_microkernel,  {0}, 0, 0, 0, 0, 0},
        {"Packed (Goto)",    gemm_packed,       {0}, 0, 0, 0, 0, 0},
        {"OMP Packed",       gemm_omp,          {0}, 0, 0, 0, 0, 0},
        {"OpenBLAS",         openblas_wrapper,  {0}, 0, 0, 0, 0, 0},
        {"BLIS",             blis_wrapper,      {0}, 0, 0, 0, 0, 0},
    };
    char openblas_name[32], blis_name_buf[32];
    snprintf(openblas_name, sizeof(openblas_name), "OpenBLAS (%dT)", nthreads);
    snprintf(blis_name_buf, sizeof(blis_name_buf), "BLIS (%dT)%s",
             nthreads, blis_dgemm_f77 ? "" : " N/A");
    impls[6].name = openblas_name;
    impls[7].name = blis_name_buf;

    const char* json_path = NULL;
    const char* preset_file = NULL;
    int custom_iter = 0, cM = 0, cN = 0, cK = 0;
    int run_all = 0;
    const char* single_preset = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            json_path = argv[++i];
        } else if (strcmp(argv[i], "--preset-file") == 0 && i + 1 < argc) {
            preset_file = argv[++i];
        } else if (strcmp(argv[i], "--custom") == 0 && i + 4 < argc) {
            custom_iter = atoi(argv[++i]);
            cM = atoi(argv[++i]);
            cN = atoi(argv[++i]);
            cK = atoi(argv[++i]);
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

    print_system_header("GEMM Benchmark");
    printf("Threads: OMP=%d, OpenBLAS=%d, BLIS=%s\n",
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
            if (!k || strcmp(k, "gemm") != 0) continue;
            int M = preset_get_int(p, "m", 0);
            int N = preset_get_int(p, "n", 0);
            int K = preset_get_int(p, "k", 0);
            int iter = preset_get_int(p, "iterations", 0);
            if (M <= 0 || N <= 0 || K <= 0 || iter <= 0) {
                fprintf(stderr, "preset '%s': m/n/k/iterations missing or invalid\n", p->name);
                continue;
            }
            run_preset(p->name, M, N, K, iter, impls, &results[num_results++]);
        }
        preset_free(head);
    } else if (custom_iter > 0) {
        results = (PresetResult*)calloc(1, sizeof(PresetResult));
        run_preset("custom", cM, cN, cK, custom_iter, impls, &results[0]);
        num_results = 1;
    } else if (run_all) {
        results = (PresetResult*)calloc(num_builtin, sizeof(PresetResult));
        for (int i = 0; i < num_builtin; i++) {
            BuiltinPreset* bp = &builtin_presets[i];
            run_preset(bp->name, bp->M, bp->N, bp->K, bp->iterations, impls, &results[i]);
        }
        num_results = num_builtin;
    } else {
        const char* name = single_preset ? single_preset : "small";
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
        run_preset(bp->name, bp->M, bp->N, bp->K, bp->iterations, impls, &results[0]);
        num_results = 1;
    }

    bench_print_summary(results, num_results, NUM_IMPLEMENTATIONS, NUM_RUNS);

    if (json_path && num_results > 0) {
        json_write_results(json_path, "gemm", nthreads, NUM_RUNS, results, num_results);
        printf("\nResults written to %s\n", json_path);
    }

cleanup:
    if (results) { bench_free_results(results, num_results); free(results); }
    blis_loader_shutdown();
    return rc;
}
