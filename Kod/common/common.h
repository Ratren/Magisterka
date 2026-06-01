#ifndef ZEN3_COMMON_H
#define ZEN3_COMMON_H

#include <stddef.h>
#include <stdio.h>

double now_seconds(void);

int    cmp_double(const void* a, const void* b);
double compute_median(double* vals, int n);
double compute_stddev(const double* vals, int n, double mean);

void   read_cpu_name(char* buf, size_t buf_size);
int    read_cache_size_bytes(int index);
double read_cpu_freq_mhz(void);
int    read_boost_state(void);
void   print_system_header(const char* title);

int    blis_loader_init(int nthreads);
void   blis_loader_shutdown(void);
void*  blis_loader_sym(const char* name);
int    blis_loader_ok(void);

int    libxsmm_loader_init(void);
void   libxsmm_loader_shutdown(void);
void*  libxsmm_loader_sym(const char* name);
int    libxsmm_loader_ok(void);

typedef struct PresetKV {
    char* key;
    char* value;
    struct PresetKV* next;
} PresetKV;

typedef struct Preset {
    char* name;
    PresetKV* kvs;
    struct Preset* next;
} Preset;

Preset*     preset_load(const char* path);
void        preset_free(Preset* head);
const char* preset_get(const Preset* p, const char* key);
long        preset_get_long(const Preset* p, const char* key, long def);
int         preset_get_int(const Preset* p, const char* key, int def);
int         preset_count(const Preset* head);

typedef struct {
    char   name[64];
    double median;
    double min;
    double max;
    double stddev;
    double max_error;
} ImplResult;

typedef struct {
    char        name[64];
    char        params_json[512];
    ImplResult* impls;
    int         num_impls;
} PresetResult;

void json_write_results(const char* path, const char* kernel,
                        int threads, int runs,
                        const PresetResult* presets, int num_presets);

typedef void (*bench_fn)(void* ctx);

typedef struct {
    double median;
    double min;
    double max;
    double stddev;
    double per_run[16];
} RunStats;

RunStats run_kernel(bench_fn fn, void* ctx,
                    int warmup_iters, int timed_iters,
                    int num_runs, double work_per_iter);

void   gen_double_arr(double* arr, size_t n);
void   gen_float_arr(float* arr, size_t n);
double max_abs_diff_d(const double* a, const double* b, size_t n);
double max_abs_diff_f(const float* a, const float* b, size_t n);

void bench_snapshot(PresetResult* out, const char* name,
                    const char* params_json,
                    const ImplResult* impls, int num_impls);
void bench_free_results(PresetResult* results, int n);
void bench_print_summary(const PresetResult* results, int n,
                         int num_impls, int num_runs);

#endif
