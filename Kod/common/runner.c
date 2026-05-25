#include "common.h"

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
