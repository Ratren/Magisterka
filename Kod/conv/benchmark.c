#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "common.h"
#include "conv.h"
#include "openblas/cblas.h"

extern void openblas_set_num_threads(int num_threads);
extern int  openblas_get_num_threads(void);
extern void goto_set_num_threads(int num_threads);

typedef void (*blis_sgemm_f77_func)(const char*, const char*,
                                    const int*, const int*, const int*,
                                    const float*, const float*, const int*,
                                    const float*, const int*,
                                    const float*, float*, const int*);
blis_sgemm_f77_func conv_blis_sgemm_f77 = NULL;

#define NUM_IMPLEMENTATIONS 11
#define NUM_RUNS 5
#define WARMUP_ITERATIONS 3
#define BUDGET_PER_RUN_SEC 1.0
#define MAX_SECONDS_PER_CALL 20.0

typedef void (*conv_func)(int, int, int, int, int, int,
                          const float*, const float*, float*);

typedef struct {
    const char* name;
    conv_func   func;
    double      runs[NUM_RUNS];
    double      median, min, max, stddev, max_error;
} Impl;

typedef struct {
    const char* name;
    int Cin, H, W, K, Cout;
    int iterations;
} BuiltinPreset;

static BuiltinPreset builtin_presets[] = {
    {"tiny",      16,  32,  32,  3, 16,  500},
    {"small",     64,  56,  56,  3, 64,  100},
    {"mid",       128, 28,  28,  3, 128, 50},
    {"large",     256, 14,  14,  3, 256, 50},
    {"xlarge",    512, 14,  14,  3, 512, 20},
    {"pointwise", 128, 56,  56,  1, 128, 50},
    {"kernel5",   32,  64,  64,  5, 32,  50},
    {"kernel7",   64,  112, 112, 7, 64,  20},
    {"rgb3x3",    3,   224, 224, 3, 64,  20},
    {"rgb5x5",    3,   128, 128, 5, 32,  50},
    {"rgb7x7",    3,   120, 120, 7, 64,  20},
    {"rgb_fhd",   3,   1080, 1920, 3, 64, 10},
};
static const int num_builtin = sizeof(builtin_presets) / sizeof(builtin_presets[0]);

static double conv_flops(int Cin, int H, int W, int KH, int KW, int Cout) {
    int OH = H - KH + 1;
    int OW = W - KW + 1;
    return 2.0 * (double)Cout * OH * OW * Cin * KH * KW;
}

static int measure_one(Impl* impl, int Cin, int H, int W, int K, int Cout,
                       int iterations, const float* X, const float* Wk,
                       float* Y, const float* Y_ref, size_t ySize) {
    double work = conv_flops(Cin, H, W, K, K, Cout);
    if (impl->median > 0.0) {
        double predicted = work / (impl->median * 1e9);
        if (predicted > MAX_SECONDS_PER_CALL) {
            impl->median = impl->min = impl->max = impl->stddev = 0.0;
            impl->max_error = 0.0;
            return -1;
        }
    }

    double t_cal = now_seconds();
    impl->func(Cin, H, W, K, K, Cout, X, Wk, Y);
    double t_one = now_seconds() - t_cal;

    if (t_one > MAX_SECONDS_PER_CALL) {
        impl->median = impl->min = impl->max = impl->stddev = 0.0;
        impl->max_error = max_abs_diff_f(Y, Y_ref, ySize);
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
        impl->func(Cin, H, W, K, K, Cout, X, Wk, Y);

    double min_g = 1e30, max_g = 0.0;
    for (int run = 0; run < runs; run++) {
        double t0 = now_seconds();
        for (int i = 0; i < iters; i++)
            impl->func(Cin, H, W, K, K, Cout, X, Wk, Y);
        double dt = now_seconds() - t0;
        double gflops = (work * iters) / (dt * 1e9);
        impl->runs[run] = gflops;
        if (gflops < min_g) min_g = gflops;
        if (gflops > max_g) max_g = gflops;
    }
    for (int r = runs; r < NUM_RUNS; r++) impl->runs[r] = impl->runs[0];

    impl->median = compute_median(impl->runs, runs);
    impl->min = min_g;
    impl->max = max_g;
    impl->stddev = (runs > 1) ? compute_stddev(impl->runs, runs, impl->median) : 0.0;
    impl->max_error = max_abs_diff_f(Y, Y_ref, ySize);
    return 1;
}

static void run_case(int Cin, int H, int W, int K, int Cout, int iterations,
                     Impl* impls) {
    int OH = H - K + 1;
    int OW = W - K + 1;
    size_t xSize = (size_t)Cin * H * W;
    size_t wSize = (size_t)Cout * Cin * K * K;
    size_t ySize = (size_t)Cout * OH * OW;

    float* X     = (float*)aligned_alloc(64, xSize * sizeof(float));
    float* Wk    = (float*)aligned_alloc(64, wSize * sizeof(float));
    float* Y     = (float*)aligned_alloc(64, ySize * sizeof(float));
    float* Y_ref = (float*)aligned_alloc(64, ySize * sizeof(float));
    if (!X || !Wk || !Y || !Y_ref) { fprintf(stderr, "alloc failed\n"); exit(1); }

    srand(42);
    gen_float_arr(X, xSize);
    gen_float_arr(Wk, wSize);

    memset(Y_ref, 0, ySize * sizeof(float));
    conv_im2col_openblas(Cin, H, W, K, K, Cout, X, Wk, Y_ref);

    for (int i = 0; i < NUM_IMPLEMENTATIONS; i++) {
        printf("  [%d/%d] %-22s", i + 1, NUM_IMPLEMENTATIONS, impls[i].name);
        fflush(stdout);
        if (impls[i].func == conv_im2col_blis && !conv_blis_sgemm_f77) {
            printf(" SKIPPED (AOCL-BLAS not loaded)\n");
            continue;
        }
        memset(Y, 0, ySize * sizeof(float));
        int ok = measure_one(&impls[i], Cin, H, W, K, Cout, iterations,
                             X, Wk, Y, Y_ref, ySize);
        if (ok == -1) {
            printf(" SKIPPED (predicted > %.0fs)\n", (double)MAX_SECONDS_PER_CALL);
        } else if (ok == 0) {
            printf(" SKIPPED (single call > %.0fs)\n", (double)MAX_SECONDS_PER_CALL);
        } else {
            printf(" med=%.2f [%.2f-%.2f] sd=%.2f err=%.1e\n",
                   impls[i].median, impls[i].min, impls[i].max,
                   impls[i].stddev, impls[i].max_error);
        }
    }

    free(X); free(Wk); free(Y); free(Y_ref);
}

static void print_table(const Impl* impls) {
    double naive_g = impls[0].median;
    double blas_g  = 0.0;
    for (int i = 0; i < NUM_IMPLEMENTATIONS; i++)
        if (strstr(impls[i].name, "OpenBLAS")) blas_g = impls[i].median;
    printf("\n---------------------------------------------------------------------------------------\n");
    printf("%-22s | %6s | %6s | %11s | %8s | %7s | %s\n",
           "Implementacja", "Median", "StdDev", "Min-Max", "vs Naive", "vs BLAS", "Blad");
    printf("-----------------------|--------|--------|-------------|----------|---------|----------\n");
    for (int i = 0; i < NUM_IMPLEMENTATIONS; i++) {
        printf("%-22s | %6.2f | %6.2f | %5.2f-%-5.2f | %7.2fx | %6.2fx | %.1e\n",
               impls[i].name, impls[i].median, impls[i].stddev,
               impls[i].min, impls[i].max,
               naive_g > 0 ? impls[i].median / naive_g : 0.0,
               blas_g  > 0 ? impls[i].median / blas_g  : 0.0,
               impls[i].max_error);
    }
    printf("---------------------------------------------------------------------------------------\n");
}

static void snapshot(PresetResult* out, const char* name,
                     int Cin, int H, int W, int K, int Cout,
                     const Impl* impls) {
    char params[128];
    snprintf(params, sizeof(params),
             "\"cin\": %d, \"h\": %d, \"w\": %d, \"k\": %d, \"cout\": %d",
             Cin, H, W, K, Cout);
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

static void run_preset(const char* name, int Cin, int H, int W, int K, int Cout,
                       int iterations, Impl* impls, PresetResult* out) {
    int OH = H - K + 1, OW = W - K + 1;
    printf("\n[ Preset: %s | C=%d HxW=%dx%d K=%d OC=%d -> %dx%d | %d iteracji ]\n\n",
           name, Cin, H, W, K, Cout, OH, OW, iterations);
    printf("Uruchamianie benchmarkow...\n");
    run_case(Cin, H, W, K, Cout, iterations, impls);
    print_table(impls);
    if (out) snapshot(out, name, Cin, H, W, K, Cout, impls);
}

static int run_measure(const char* name, int Cin, int H, int W, int K, int Cout,
                       int iterations, Impl* impls) {
    if (Cin <= 0 || H <= 0 || W <= 0 || K <= 0 || Cout <= 0 || iterations <= 0) {
        fprintf(stderr, "run_measure: wymagane --custom <iter> <Cin> <H> <W> <K> <Cout>\n");
        return 1;
    }
    int idx = -1;
    for (int i = 0; i < NUM_IMPLEMENTATIONS; i++)
        if (strcmp(impls[i].name, name) == 0) { idx = i; break; }
    if (idx < 0) { fprintf(stderr, "run_measure: nieznana implementacja '%s'\n", name); return 1; }
    if (impls[idx].func == conv_im2col_blis && !conv_blis_sgemm_f77) {
        fprintf(stderr, "run_measure: '%s' niedostepna (AOCL-BLAS)\n", name);
        return 2;
    }
    if (K > H || K > W) {
        fprintf(stderr, "run_measure: K (%d) przekracza H (%d) lub W (%d)\n", K, H, W);
        return 1;
    }
    int OH = H - K + 1, OW = W - K + 1;
    size_t xSize = (size_t)Cin * H * W;
    size_t wSize = (size_t)Cout * Cin * K * K;
    size_t ySize = (size_t)Cout * OH * OW;
    float* X  = (float*)aligned_alloc(64, xSize * sizeof(float));
    float* Wk = (float*)aligned_alloc(64, wSize * sizeof(float));
    float* Y  = (float*)aligned_alloc(64, ySize * sizeof(float));
    if (!X || !Wk || !Y) { fprintf(stderr, "alloc failed\n"); return 1; }
    srand(42);
    gen_float_arr(X, xSize);
    gen_float_arr(Wk, wSize);
    memset(Y, 0, ySize * sizeof(float));
    for (int w = 0; w < WARMUP_ITERATIONS; w++) impls[idx].func(Cin, H, W, K, K, Cout, X, Wk, Y);
    for (int it = 0; it < iterations; it++) impls[idx].func(Cin, H, W, K, K, Cout, X, Wk, Y);
    free(X); free(Wk); free(Y);
    return 0;
}

static void print_usage(const char* prog) {
    printf("Uzycie:\n");
    printf("  %s                                       - uruchamia preset 'small'\n", prog);
    printf("  %s <preset>                              - wbudowany preset\n", prog);
    printf("  %s all                                   - wszystkie wbudowane presety\n", prog);
    printf("  %s --preset-file <path>                  - presety z pliku INI (kernel = conv)\n", prog);
    printf("  %s --custom <iter> <Cin> <H> <W> <K> <Cout>\n", prog);
    printf("  %s [--json <path>] ...                   - zapisz wyniki do JSON\n", prog);
    printf("\nWbudowane presety:\n");
    for (int i = 0; i < num_builtin; i++) {
        printf("  %-10s - C=%d HxW=%dx%d K=%d OC=%d, %d iteracji\n",
               builtin_presets[i].name, builtin_presets[i].Cin,
               builtin_presets[i].H, builtin_presets[i].W,
               builtin_presets[i].K, builtin_presets[i].Cout,
               builtin_presets[i].iterations);
    }
}

int main(int argc, char* argv[]) {
    int nthreads = omp_get_max_threads();
    openblas_set_num_threads(nthreads);
    goto_set_num_threads(nthreads);
    int has_blis = blis_loader_init(nthreads);
    if (has_blis) conv_blis_sgemm_f77 = (blis_sgemm_f77_func)blis_loader_sym("sgemm_");

    Impl impls[NUM_IMPLEMENTATIONS] = {
        {"Naive",              conv_naive,           {0}, 0, 0, 0, 0, 0},
        {"Loop Reorder",       conv_reorder,         {0}, 0, 0, 0, 0, 0},
        {"Blocked",            conv_blocked,         {0}, 0, 0, 0, 0, 0},
        {"Packed Direct",      conv_packed,          {0}, 0, 0, 0, 0, 0},
        {"OMP Packed",         conv_omp,             {0}, 0, 0, 0, 0, 0},
        {"NCHWc direct",       conv_nchwc,           {0}, 0, 0, 0, 0, 0},
        {"1x1 (SGEMM)",        conv_1x1,             {0}, 0, 0, 0, 0, 0},
        {"Winograd F(2,3)",    conv_winograd,        {0}, 0, 0, 0, 0, 0},
        {"Zen3 dispatch OMP",  conv_zen3_omp,        {0}, 0, 0, 0, 0, 0},
        {"im2col + OpenBLAS",  conv_im2col_openblas, {0}, 0, 0, 0, 0, 0},
        {"im2col + AOCL-BLAS",      conv_im2col_blis,     {0}, 0, 0, 0, 0, 0},
    };
    char openblas_name[40], blis_name_buf[40];
    snprintf(openblas_name, sizeof(openblas_name), "im2col + OpenBLAS (%dT)", nthreads);
    snprintf(blis_name_buf, sizeof(blis_name_buf), "im2col + AOCL-BLAS (%dT)%s",
             nthreads, conv_blis_sgemm_f77 ? "" : " N/A");
    impls[9].name = openblas_name;
    impls[10].name = blis_name_buf;

    const char* json_path = NULL;
    const char* preset_file = NULL;
    int custom_iter = 0, cCin = 0, cH = 0, cW = 0, cK = 0, cCout = 0;
    int run_all = 0;
    const char* single_preset = NULL;
    const char* measure_impl = NULL;
    int list_impls = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            json_path = argv[++i];
        } else if (strcmp(argv[i], "--preset-file") == 0 && i + 1 < argc) {
            preset_file = argv[++i];
        } else if (strcmp(argv[i], "--custom") == 0 && i + 6 < argc) {
            custom_iter = atoi(argv[++i]);
            cCin = atoi(argv[++i]); cH = atoi(argv[++i]); cW = atoi(argv[++i]);
            cK = atoi(argv[++i]); cCout = atoi(argv[++i]);
        } else if (strcmp(argv[i], "all") == 0) {
            run_all = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            blis_loader_shutdown();
            return 0;
        } else if (strcmp(argv[i], "--measure") == 0 && i + 1 < argc) {
            measure_impl = argv[++i];
        } else if (strcmp(argv[i], "--list-impls") == 0) {
            list_impls = 1;
        } else {
            single_preset = argv[i];
        }
    }

    if (list_impls) {
        for (int i = 0; i < NUM_IMPLEMENTATIONS; i++) printf("%s\n", impls[i].name);
        blis_loader_shutdown();
        return 0;
    }
    if (measure_impl) {
        int mrc = run_measure(measure_impl, cCin, cH, cW, cK, cCout, custom_iter, impls);
        blis_loader_shutdown();
        return mrc;
    }

    print_system_header("Convolution Benchmark");
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
            if (!k || strcmp(k, "conv") != 0) continue;
            int Cin = preset_get_int(p, "cin", 0);
            int H   = preset_get_int(p, "h", 0);
            int W   = preset_get_int(p, "w", 0);
            int K   = preset_get_int(p, "k", 0);
            int Cout= preset_get_int(p, "cout", 0);
            int iter= preset_get_int(p, "iterations", 0);
            if (Cin <= 0 || H <= 0 || W <= 0 || K <= 0 || Cout <= 0 || iter <= 0) {
                fprintf(stderr, "preset '%s': cin/h/w/k/cout/iterations missing\n", p->name);
                continue;
            }
            run_preset(p->name, Cin, H, W, K, Cout, iter, impls, &results[num_results++]);
        }
        preset_free(head);
    } else if (custom_iter > 0) {
        results = (PresetResult*)calloc(1, sizeof(PresetResult));
        run_preset("custom", cCin, cH, cW, cK, cCout, custom_iter, impls, &results[0]);
        num_results = 1;
    } else if (run_all) {
        results = (PresetResult*)calloc(num_builtin, sizeof(PresetResult));
        for (int i = 0; i < num_builtin; i++) {
            BuiltinPreset* bp = &builtin_presets[i];
            run_preset(bp->name, bp->Cin, bp->H, bp->W, bp->K, bp->Cout,
                       bp->iterations, impls, &results[i]);
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
        run_preset(bp->name, bp->Cin, bp->H, bp->W, bp->K, bp->Cout,
                   bp->iterations, impls, &results[0]);
        num_results = 1;
    }

    bench_print_summary(results, num_results, NUM_IMPLEMENTATIONS, NUM_RUNS);

    if (json_path && num_results > 0) {
        json_write_results(json_path, "conv", nthreads, NUM_RUNS, results, num_results);
        printf("\nResults written to %s\n", json_path);
    }

cleanup:
    if (results) { bench_free_results(results, num_results); free(results); }
    blis_loader_shutdown();
    return rc;
}
