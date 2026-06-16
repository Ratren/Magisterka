#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void bench_snapshot(PresetResult* out, const char* name,
                    const char* params_json,
                    const ImplResult* impls, int num_impls) {
    strncpy(out->name, name, sizeof(out->name) - 1);
    out->name[sizeof(out->name) - 1] = 0;
    strncpy(out->params_json, params_json, sizeof(out->params_json) - 1);
    out->params_json[sizeof(out->params_json) - 1] = 0;
    out->num_impls = num_impls;
    out->impls = (ImplResult*)calloc(num_impls, sizeof(ImplResult));
    for (int i = 0; i < num_impls; i++) out->impls[i] = impls[i];
}

void bench_free_results(PresetResult* results, int n) {
    for (int i = 0; i < n; i++) free(results[i].impls);
}

void bench_print_summary(const PresetResult* results, int n,
                         int num_impls, int num_runs) {
    if (n <= 1) return;
    printf("\n================================================================\n");
    printf("              PODSUMOWANIE - Median GFLOPS (%d runs)\n", num_runs);
    printf("================================================================\n");
    printf("%-22s |", "Implementacja");
    for (int p = 0; p < n; p++) printf(" %10s |", results[p].name);
    printf("\n-----------------------|");
    for (int p = 0; p < n; p++) printf("------------|");
    printf("\n");
    for (int i = 0; i < num_impls; i++) {
        printf("%-22s |", results[0].impls[i].name);
        for (int p = 0; p < n; p++) printf(" %10.2f |", results[p].impls[i].median);
        printf("\n");
    }
    printf("================================================================\n");
}

RunStats run_kernel(bench_fn fn, void* ctx,
                    int warmup_iters, int timed_iters,
                    int num_runs, double work_per_iter) {
    RunStats s = {0};
    if (num_runs > 16) num_runs = 16;

    for (int w = 0; w < warmup_iters; w++) fn(ctx);

    double min_g = 1e30, max_g = 0.0;
    for (int run = 0; run < num_runs; run++) {
        double t0 = now_seconds();
        for (int i = 0; i < timed_iters; i++) fn(ctx);
        double dt = now_seconds() - t0;
        double gflops = (work_per_iter * (double)timed_iters) / (dt * 1e9);
        s.per_run[run] = gflops;
        if (gflops < min_g) min_g = gflops;
        if (gflops > max_g) max_g = gflops;
    }

    double tmp[16];
    for (int i = 0; i < num_runs; i++) tmp[i] = s.per_run[i];
    s.median = compute_median(tmp, num_runs);
    s.min = min_g;
    s.max = max_g;
    s.stddev = compute_stddev(s.per_run, num_runs, s.median);
    return s;
}
