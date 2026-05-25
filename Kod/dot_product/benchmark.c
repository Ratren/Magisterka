#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "common.h"
#include "dot.h"
#include "openblas/cblas.h"

extern void openblas_set_num_threads(int num_threads);
extern int  openblas_get_num_threads(void);
extern void goto_set_num_threads(int num_threads);

#define NUM_IMPLEMENTATIONS 6
#define NUM_RUNS 5
#define WARMUP_ITERATIONS 10

typedef double (*dot_func)(const double*, const double*, size_t);

typedef struct {
    const char* name;
    dot_func    func;
    double      runs[NUM_RUNS];
    double      median;
    double      min;
    double      max;
    double      stddev;
    double      abs_error;
} Impl;

typedef struct {
    const char* name;
    size_t      n;
    int         iterations;
} BuiltinPreset;

static BuiltinPreset builtin_presets[] = {
    {"l1_fit",  2048,        5000000},
    {"l2_fit",  32768,       300000},
    {"l3_fit",  2097152,     2000},
    {"dram",    67108864,    50},
};
static const int num_builtin = sizeof(builtin_presets) / sizeof(builtin_presets[0]);

typedef double (*blis_ddot_f77_func)(const int*, const double*, const int*,
                                     const double*, const int*);
static blis_ddot_f77_func blis_ddot_f77 = NULL;

static double openblas_wrapper(const double* a, const double* b, size_t n) {
    return cblas_ddot((int)n, a, 1, b, 1);
}

static double blis_wrapper(const double* a, const double* b, size_t n) {
    if (!blis_ddot_f77) return 0.0;
    int nn = (int)n, inc = 1;
    return blis_ddot_f77(&nn, a, &inc, b, &inc);
}

static void gen_double_arr(double* arr, size_t n) {
    for (size_t i = 0; i < n; i++) arr[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

#define BUDGET_PER_RUN_SEC 1.0
#define MAX_SECONDS_PER_CALL 20.0

static int measure_one(Impl* impl, const double* a, const double* b,
                       size_t n, int iterations, double ref) {
    double work = 2.0 * (double)n;
    if (impl->median > 0.0) {
        double predicted = work / (impl->median * 1e9);
        if (predicted > MAX_SECONDS_PER_CALL) {
            impl->median = impl->min = impl->max = impl->stddev = 0.0;
            impl->abs_error = 0.0;
            return -1;
        }
    }

    double t_cal = now_seconds();
    double last = impl->func(a, b, n);
    double t_one = now_seconds() - t_cal;

    if (t_one > MAX_SECONDS_PER_CALL) {
        impl->median = impl->min = impl->max = impl->stddev = 0.0;
        impl->abs_error = fabs(last - ref);
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
    for (int w = 0; w < warmup; w++) last = impl->func(a, b, n);

    double total_flops = 2.0 * (double)n;
    double min_g = 1e30, max_g = 0.0;

    for (int run = 0; run < runs; run++) {
        double t0 = now_seconds();
        for (int i = 0; i < iters; i++) last = impl->func(a, b, n);
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
    impl->abs_error = fabs(last - ref);
    return 1;
}

static void run_case(size_t n, int iterations, Impl* impls) {
    double* a = (double*)aligned_alloc(64, n * sizeof(double));
    double* b = (double*)aligned_alloc(64, n * sizeof(double));
    if (!a || !b) { fprintf(stderr, "alloc failed (n=%zu)\n", n); exit(1); }

    srand(42);
    gen_double_arr(a, n);
    gen_double_arr(b, n);

    double ref = openblas_wrapper(a, b, n);

    for (int i = 0; i < NUM_IMPLEMENTATIONS; i++) {
        printf("  [%d/%d] %-20s", i + 1, NUM_IMPLEMENTATIONS, impls[i].name);
        fflush(stdout);
        if (impls[i].func == blis_wrapper && !blis_ddot_f77) {
            printf(" SKIPPED (BLIS not loaded)\n");
            continue;
        }
        int ok = measure_one(&impls[i], a, b, n, iterations, ref);
        if (ok == -1) {
            printf(" SKIPPED (predicted > %.0fs based on prior preset)\n", (double)MAX_SECONDS_PER_CALL);
        } else if (ok == 0) {
            printf(" SKIPPED (single call > %.0fs)\n", (double)MAX_SECONDS_PER_CALL);
        } else {
            printf(" med=%.2f [%.2f-%.2f] sd=%.2f\n",
                   impls[i].median, impls[i].min, impls[i].max, impls[i].stddev);
        }
    }

    free(a); free(b);
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
               impls[i].median / naive_g,
               impls[i].median / blas_g,
               impls[i].abs_error);
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

static void snapshot(PresetResult* out, const char* name, size_t n, const Impl* impls) {
    strncpy(out->name, name, sizeof(out->name) - 1);
    out->name[sizeof(out->name) - 1] = 0;
    snprintf(out->params_json, sizeof(out->params_json), "\"size\": %zu", n);
    out->num_impls = NUM_IMPLEMENTATIONS;
    out->impls = (ImplResult*)calloc(NUM_IMPLEMENTATIONS, sizeof(ImplResult));
    for (int i = 0; i < NUM_IMPLEMENTATIONS; i++) {
        strncpy(out->impls[i].name, impls[i].name, sizeof(out->impls[i].name) - 1);
        out->impls[i].median = impls[i].median;
        out->impls[i].min = impls[i].min;
        out->impls[i].max = impls[i].max;
        out->impls[i].stddev = impls[i].stddev;
        out->impls[i].max_error = impls[i].abs_error;
    }
}

static void run_preset(const char* name, size_t n, int iterations,
                       Impl* impls, PresetResult* out) {
    printf("\n[ Preset: %s | n=%zu | %d iteracji ]\n\n", name, n, iterations);
    printf("Uruchamianie benchmarkow...\n");
    run_case(n, iterations, impls);
    print_table(impls);
    if (out) snapshot(out, name, n, impls);
}

static void print_summary(const PresetResult* results, int n) {
    if (n <= 1) return;
    printf("\n================================================================\n");
    printf("              PODSUMOWANIE - Median GFLOPS (%d runs)\n", NUM_RUNS);
    printf("================================================================\n");
    printf("%-20s |", "Implementacja");
    for (int p = 0; p < n; p++) printf(" %8s |", results[p].name);
    printf("\n---------------------|");
    for (int p = 0; p < n; p++) printf("----------|");
    printf("\n");
    for (int i = 0; i < NUM_IMPLEMENTATIONS; i++) {
        printf("%-20s |", results[0].impls[i].name);
        for (int p = 0; p < n; p++) printf(" %8.2f |", results[p].impls[i].median);
        printf("\n");
    }
    printf("================================================================\n");
}

static void free_results(PresetResult* results, int n) {
    for (int i = 0; i < n; i++) free(results[i].impls);
}

static void print_usage(const char* prog) {
    printf("Uzycie:\n");
    printf("  %s                          - uruchamia preset 'l2_fit'\n", prog);
    printf("  %s <preset>                 - uruchamia wbudowany preset\n", prog);
    printf("  %s all                      - uruchamia wszystkie wbudowane presety\n", prog);
    printf("  %s --preset-file <path>     - presety z pliku INI (kernel = dot)\n", prog);
    printf("  %s --custom <iter> <size>   - parametry recznie\n", prog);
    printf("  %s [--json <path>] ...      - zapisz wyniki do JSON\n", prog);
    printf("\nWbudowane presety:\n");
    for (int i = 0; i < num_builtin; i++) {
        printf("  %-8s - n=%zu, %d iteracji\n",
               builtin_presets[i].name, builtin_presets[i].n,
               builtin_presets[i].iterations);
    }
}

int main(int argc, char* argv[]) {
    int nthreads = omp_get_max_threads();
    openblas_set_num_threads(nthreads);
    goto_set_num_threads(nthreads);
    int has_blis = blis_loader_init(nthreads);
    if (has_blis) blis_ddot_f77 = (blis_ddot_f77_func)blis_loader_sym("ddot_");

    Impl impls[NUM_IMPLEMENTATIONS] = {
        {"Naive",         dot_naive,         {0}, 0, 0, 0, 0, 0},
        {"SIMD",          dot_simd,          {0}, 0, 0, 0, 0, 0},
        {"SIMD MultiAcc", dot_simd_multiacc, {0}, 0, 0, 0, 0, 0},
        {"OMP",           dot_omp,           {0}, 0, 0, 0, 0, 0},
        {"OpenBLAS",      openblas_wrapper,  {0}, 0, 0, 0, 0, 0},
        {"BLIS",          blis_wrapper,      {0}, 0, 0, 0, 0, 0},
    };
    char openblas_name[32], blis_name_buf[32];
    snprintf(openblas_name, sizeof(openblas_name), "OpenBLAS (%dT)", nthreads);
    snprintf(blis_name_buf, sizeof(blis_name_buf), "BLIS (%dT)%s",
             nthreads, blis_ddot_f77 ? "" : " N/A");
    impls[4].name = openblas_name;
    impls[5].name = blis_name_buf;

    const char* json_path = NULL;
    const char* preset_file = NULL;
    int custom_iter = 0;
    size_t custom_n = 0;
    int run_all = 0;
    const char* single_preset = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            json_path = argv[++i];
        } else if (strcmp(argv[i], "--preset-file") == 0 && i + 1 < argc) {
            preset_file = argv[++i];
        } else if (strcmp(argv[i], "--custom") == 0 && i + 2 < argc) {
            custom_iter = atoi(argv[++i]);
            custom_n = (size_t)strtoull(argv[++i], NULL, 10);
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

    print_system_header("Dot Product Benchmark");
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
            if (!k || strcmp(k, "dot") != 0) continue;
            size_t n = (size_t)preset_get_long(p, "size", 0);
            int iter = preset_get_int(p, "iterations", 0);
            if (n == 0 || iter <= 0) {
                fprintf(stderr, "preset '%s': size/iterations missing or invalid\n", p->name);
                continue;
            }
            run_preset(p->name, n, iter, impls, &results[num_results++]);
        }
        preset_free(head);
    } else if (custom_iter > 0) {
        results = (PresetResult*)calloc(1, sizeof(PresetResult));
        run_preset("custom", custom_n, custom_iter, impls, &results[0]);
        num_results = 1;
    } else if (run_all) {
        results = (PresetResult*)calloc(num_builtin, sizeof(PresetResult));
        for (int i = 0; i < num_builtin; i++) {
            BuiltinPreset* bp = &builtin_presets[i];
            run_preset(bp->name, bp->n, bp->iterations, impls, &results[i]);
        }
        num_results = num_builtin;
    } else {
        const char* name = single_preset ? single_preset : "l2_fit";
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
        run_preset(bp->name, bp->n, bp->iterations, impls, &results[0]);
        num_results = 1;
    }

    print_summary(results, num_results);

    if (json_path && num_results > 0) {
        json_write_results(json_path, "dot", nthreads, NUM_RUNS, results, num_results);
        printf("\nResults written to %s\n", json_path);
    }

cleanup:
    if (results) { free_results(results, num_results); free(results); }
    blis_loader_shutdown();
    return rc;
}
