#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <ctype.h>
#include <dlfcn.h>
#include <omp.h>
#include "gemv.h"
#include "openblas/cblas.h"

extern void openblas_set_num_threads(int num_threads);
extern int openblas_get_num_threads(void);
extern void goto_set_num_threads(int num_threads);

#define NUM_IMPLEMENTATIONS 9
#define NUM_RUNS 5
#define WARMUP_ITERATIONS 10

static const char* json_output_path = NULL;

typedef struct {
    double median;
    double min;
    double max;
    double stddev;
} PerfResult;

static PerfResult saved_results[6][NUM_IMPLEMENTATIONS];

static void* blis_handle = NULL;
typedef void (*blis_dgemv_f77_func)(const char* trans, const int* m, const int* n,
                                    const double* alpha, const double* A, const int* lda,
                                    const double* x, const int* incx,
                                    const double* beta, double* y, const int* incy);
typedef void (*blis_set_threads_func)(int);
static blis_dgemv_f77_func blis_dgemv_f77 = NULL;
static blis_set_threads_func blis_set_num_threads = NULL;

typedef void (*gemv_func)(int, int, double, const double*, const double*, double, double*);

typedef struct {
    const char* name;
    gemv_func func;
    double gflops_runs[NUM_RUNS];
    double median_gflops;
    double min_gflops;
    double max_gflops;
    double stddev_gflops;
    double max_error;
} Implementation;

typedef struct {
    const char* name;
    int rows;
    int cols;
    int iterations;
} Preset;

static Preset presets[] = {
    {"tiny",   64,   64,   5000000},
    {"small",  256,  256,  200000},
    {"medium", 1024, 1024, 10000},
    {"large",  4096, 4096, 300},
    {"wide",   256,  8192, 5000},
    {"tall",   8192, 256,  5000},
};
static const int num_presets = sizeof(presets) / sizeof(presets[0]);

static void openblas_wrapper(int rows, int cols, double alpha,
                             const double* A, const double* x,
                             double beta, double* y) {
    cblas_dgemv(CblasRowMajor, CblasNoTrans, rows, cols, alpha, A, cols, x, 1, beta, y, 1);
}

static void blis_wrapper(int rows, int cols, double alpha,
                          const double* A, const double* x,
                          double beta, double* y) {
    if (blis_dgemv_f77) {
        // Fortran dgemv_ is column-major. For row-major A*x we use:
        // In column-major view, our row-major MxN matrix is an NxM matrix,
        // so we pass trans="T", m=cols, n=rows, lda=cols
        char trans = 'T';
        int m = cols, n = rows, lda = cols, incx = 1, incy = 1;
        blis_dgemv_f77(&trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
    }
}

static int load_blis(int nthreads) {
    const char* paths[] = {
        "libblis-mt.so",
        "../blis_install/lib/libblis-mt.so",
        "/usr/local/lib/libblis-mt.so",
        NULL
    };

    for (int i = 0; paths[i]; i++) {
        blis_handle = dlopen(paths[i], RTLD_NOW | RTLD_LOCAL);
        if (blis_handle) break;
    }

    if (!blis_handle) {
        fprintf(stderr, "BLIS nie znaleziony: %s\n", dlerror());
        return 0;
    }

    blis_dgemv_f77 = (blis_dgemv_f77_func)dlsym(blis_handle, "dgemv_");
    if (!blis_dgemv_f77) {
        fprintf(stderr, "BLIS: brak dgemv_: %s\n", dlerror());
        dlclose(blis_handle);
        blis_handle = NULL;
        return 0;
    }

    blis_set_num_threads = (blis_set_threads_func)dlsym(blis_handle, "bli_thread_set_num_threads");
    if (blis_set_num_threads) {
        blis_set_num_threads(nthreads);
    }

    return 1;
}

static int read_cache_size_bytes(int index) {
    char path[128];
    char size_str[32];

    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu0/cache/index%d/size", index);
    FILE *fp = fopen(path, "r");

    if (!fp) return -1;

    if (!fgets(size_str, sizeof(size_str), fp)) {
        fclose(fp);
        return -1;
    }
    fclose(fp);

    size_str[strcspn(size_str, "\n")] = 0;

    long size = strtol(size_str, NULL, 10);
    char unit = size_str[strlen(size_str) - 1];

    switch (toupper(unit)) {
        case 'K': size *= 1024; break;
        case 'M': size *= 1024 * 1024; break;
    }

    return (int)size;
}

static void read_cpu_name(char* buf, size_t buf_size) {
    FILE* fp = fopen("/proc/cpuinfo", "r");
    if (!fp) {
        strncpy(buf, "Unknown CPU", buf_size);
        return;
    }

    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "model name", 10) == 0) {
            char* colon = strchr(line, ':');
            if (colon) {
                colon++;
                while (*colon == ' ' || *colon == '\t') colon++;
                colon[strcspn(colon, "\n")] = 0;
                strncpy(buf, colon, buf_size);
                fclose(fp);
                return;
            }
        }
    }
    fclose(fp);
    strncpy(buf, "Unknown CPU", buf_size);
}

static double read_cpu_freq_mhz(void) {
    // Try scaling_cur_freq first (actual current frequency)
    FILE* fp = fopen("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "r");
    if (!fp) {
        // Fallback to /proc/cpuinfo
        fp = fopen("/proc/cpuinfo", "r");
        if (!fp) return -1;
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            if (strncmp(line, "cpu MHz", 7) == 0) {
                char* colon = strchr(line, ':');
                if (colon) {
                    fclose(fp);
                    return atof(colon + 1);
                }
            }
        }
        fclose(fp);
        return -1;
    }
    char buf[32];
    if (fgets(buf, sizeof(buf), fp)) {
        fclose(fp);
        return atof(buf) / 1000.0; 
    }
    fclose(fp);
    return -1;
}

static int read_boost_state(void) {
    FILE* fp = fopen("/sys/devices/system/cpu/cpufreq/boost", "r");
    if (!fp) return -1;
    char buf[8];
    if (fgets(buf, sizeof(buf), fp)) {
        fclose(fp);
        return atoi(buf);
    }
    fclose(fp);
    return -1;
}

static void gen_double_arr(double* arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
}

static double calc_max_error(const double* y1, const double* y2, int size) {
    double max_err = 0.0;
    for (int i = 0; i < size; i++) {
        double err = fabs(y1[i] - y2[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

static int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

static double compute_median(double* vals, int n) {
    double sorted[NUM_RUNS];
    memcpy(sorted, vals, n * sizeof(double));
    qsort(sorted, n, sizeof(double), cmp_double);
    if (n % 2 == 1) return sorted[n / 2];
    return (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
}

static double compute_stddev(double* vals, int n, double mean) {
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        double d = vals[i] - mean;
        sum_sq += d * d;
    }
    return sqrt(sum_sq / n);
}

static void run_benchmark(int rows, int cols, int iterations, Implementation* impls) {
    double* A = (double*)aligned_alloc(32, rows * cols * sizeof(double));
    double* x = (double*)aligned_alloc(32, cols * sizeof(double));
    double* y = (double*)aligned_alloc(32, rows * sizeof(double));
    double* y_ref = (double*)aligned_alloc(32, rows * sizeof(double));

    if (!A || !x || !y || !y_ref) {
        fprintf(stderr, "Blad alokacji pamieci!\n");
        exit(1);
    }

    srand(42);
    gen_double_arr(A, rows * cols);
    gen_double_arr(x, cols);

    double alpha = 1.0;
    double beta = 0.0;

    memset(y_ref, 0, rows * sizeof(double));
    openblas_wrapper(rows, cols, alpha, A, x, beta, y_ref);

    long total_flops = 2L * rows * cols;
    struct timespec start, end;

    for (int impl = 0; impl < NUM_IMPLEMENTATIONS; impl++) {
        printf("  [%d/%d] %-20s", impl + 1, NUM_IMPLEMENTATIONS, impls[impl].name);
        fflush(stdout);

        for (int w = 0; w < WARMUP_ITERATIONS; w++) {
            impls[impl].func(rows, cols, alpha, A, x, beta, y);
        }

        double min_g = 1e18, max_g = 0.0;
        for (int run = 0; run < NUM_RUNS; run++) {
            clock_gettime(CLOCK_MONOTONIC_RAW, &start);
            for (int i = 0; i < iterations; i++) {
                impls[impl].func(rows, cols, alpha, A, x, beta, y);
            }
            clock_gettime(CLOCK_MONOTONIC_RAW, &end);

            double elapsed = (end.tv_sec - start.tv_sec) +
                             (end.tv_nsec - start.tv_nsec) / 1e9;
            double gflops = (double)(total_flops * iterations) / (elapsed * 1e9);
            impls[impl].gflops_runs[run] = gflops;

            if (gflops < min_g) min_g = gflops;
            if (gflops > max_g) max_g = gflops;
        }

        impls[impl].median_gflops = compute_median(impls[impl].gflops_runs, NUM_RUNS);
        impls[impl].min_gflops = min_g;
        impls[impl].max_gflops = max_g;
        impls[impl].stddev_gflops = compute_stddev(impls[impl].gflops_runs, NUM_RUNS,
                                                    impls[impl].median_gflops);
        impls[impl].max_error = calc_max_error(y, y_ref, rows);

        printf(" med=%.2f [%.2f-%.2f] sd=%.2f\n",
               impls[impl].median_gflops,
               min_g, max_g,
               impls[impl].stddev_gflops);
    }

    free(A);
    free(x);
    free(y);
    free(y_ref);
}

static void print_results(Implementation* impls, int rows, int cols, int iterations) {
    double naive_gflops = impls[0].median_gflops;
    // OpenBLAS is at index NUM_IMPLEMENTATIONS - 2 (BLIS is last)
    double blas_gflops = impls[NUM_IMPLEMENTATIONS - 2].median_gflops;

    printf("\n");
    printf("-------------------------------------------------------------------------------------\n");
    printf("%-20s | %6s | %6s | %11s | %8s | %7s | %s\n",
           "Implementacja", "Median", "StdDev", "Min-Max", "vs Naive", "vs BLAS", "Blad");
    printf("---------------------|--------|--------|-------------|----------|---------|----------\n");

    for (int i = 0; i < NUM_IMPLEMENTATIONS; i++) {
        printf("%-20s | %6.2f | %6.2f | %5.2f-%-5.2f | %7.2fx | %6.2fx | %.1e\n",
               impls[i].name,
               impls[i].median_gflops,
               impls[i].stddev_gflops,
               impls[i].min_gflops,
               impls[i].max_gflops,
               impls[i].median_gflops / naive_gflops,
               impls[i].median_gflops / blas_gflops,
               impls[i].max_error);
    }
    printf("-------------------------------------------------------------------------------------\n");

    int best_idx = 0;
    double best_gflops = 0.0;
    // Search only our implementations (exclude OpenBLAS and BLIS)
    for (int i = 0; i < NUM_IMPLEMENTATIONS - 2; i++) {
        if (impls[i].median_gflops > best_gflops) {
            best_gflops = impls[i].median_gflops;
            best_idx = i;
        }
    }

    double vs_blas = impls[best_idx].median_gflops / blas_gflops;
    if (vs_blas >= 1.0) {
        printf("\nNajlepsza: %s (%.2fx szybsza niz OpenBLAS)\n",
               impls[best_idx].name, vs_blas);
    } else {
        printf("\nNajlepsza: %s (%.2fx wolniejsza niz OpenBLAS)\n",
               impls[best_idx].name, 1.0 / vs_blas);
    }
}

static void print_header(void) {
    char cpu_name[256];
    read_cpu_name(cpu_name, sizeof(cpu_name));

    int l1 = read_cache_size_bytes(0);
    int l2 = read_cache_size_bytes(2);
    int l3 = read_cache_size_bytes(3);
    double freq = read_cpu_freq_mhz();
    int boost = read_boost_state();

    printf("================================================================\n");
    printf("              GEMV Benchmark - %s\n", cpu_name);
    printf("================================================================\n");

    if (l1 > 0 && l2 > 0 && l3 > 0) {
        printf("L1: %d KB | L2: %d KB | L3: %d MB\n",
               l1 / 1024, l2 / 1024, l3 / (1024 * 1024));
    }
    if (freq > 0) {
        printf("CPU freq: %.0f MHz | Boost: %s\n", freq,
               boost < 0 ? "unknown" : (boost ? "ON" : "OFF"));
    }
    printf("Runs per impl: %d | Clock: MONOTONIC_RAW\n", NUM_RUNS);
    printf("Flagi kompilatora: -O3 -march=znver3 -mfma -mavx2 -ffast-math\n");
    printf("================================================================\n\n");
}

static void run_single_preset(Preset* p, Implementation* impls) {
    printf("[ Preset: %s | %dx%d | %d iteracji ]\n\n",
           p->name, p->rows, p->cols, p->iterations);
    printf("Uruchamianie benchmarkow...\n");

    run_benchmark(p->rows, p->cols, p->iterations, impls);
    print_results(impls, p->rows, p->cols, p->iterations);
}

static void write_json_results(Implementation* impls) {
    FILE* f = fopen(json_output_path, "w");
    if (!f) { fprintf(stderr, "Cannot open %s for writing\n", json_output_path); return; }

    char cpu_name[256];
    read_cpu_name(cpu_name, sizeof(cpu_name));
    int nthreads = omp_get_max_threads();

    /* escape any quotes in cpu_name just in case */
    fprintf(f, "{\n");
    fprintf(f, "  \"cpu\": \"%s\",\n", cpu_name);
    fprintf(f, "  \"threads\": %d,\n", nthreads);
    fprintf(f, "  \"runs\": %d,\n", NUM_RUNS);
    fprintf(f, "  \"presets\": [\n");

    for (int p = 0; p < num_presets; p++) {
        fprintf(f, "    {\n");
        fprintf(f, "      \"name\": \"%s\",\n", presets[p].name);
        fprintf(f, "      \"rows\": %d,\n", presets[p].rows);
        fprintf(f, "      \"cols\": %d,\n", presets[p].cols);
        fprintf(f, "      \"implementations\": {\n");
        for (int i = 0; i < NUM_IMPLEMENTATIONS; i++) {
            fprintf(f, "        \"%s\": {\"median\": %.4f, \"min\": %.4f, \"max\": %.4f, \"stddev\": %.4f}%s\n",
                    impls[i].name,
                    saved_results[p][i].median,
                    saved_results[p][i].min,
                    saved_results[p][i].max,
                    saved_results[p][i].stddev,
                    i < NUM_IMPLEMENTATIONS - 1 ? "," : "");
        }
        fprintf(f, "      }\n");
        fprintf(f, "    }%s\n", p < num_presets - 1 ? "," : "");
    }

    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    fclose(f);
    printf("\nResults written to %s\n", json_output_path);
}

static void run_all_presets(Implementation* impls) {
    double all_results[num_presets][NUM_IMPLEMENTATIONS];

    for (int p = 0; p < num_presets; p++) {
        printf("\n[ Preset: %s | %dx%d | %d iteracji ]\n\n",
               presets[p].name, presets[p].rows, presets[p].cols, presets[p].iterations);
        printf("Uruchamianie benchmarkow...\n");

        run_benchmark(presets[p].rows, presets[p].cols, presets[p].iterations, impls);
        print_results(impls, presets[p].rows, presets[p].cols, presets[p].iterations);

        for (int i = 0; i < NUM_IMPLEMENTATIONS; i++) {
            all_results[p][i] = impls[i].median_gflops;
            saved_results[p][i].median = impls[i].median_gflops;
            saved_results[p][i].min    = impls[i].min_gflops;
            saved_results[p][i].max    = impls[i].max_gflops;
            saved_results[p][i].stddev = impls[i].stddev_gflops;
        }

        printf("\n");
    }

    // Podsumowanie
    printf("\n================================================================\n");
    printf("              PODSUMOWANIE - Median GFLOPS (%d runs)\n", NUM_RUNS);
    printf("================================================================\n");

    printf("%-20s |", "Implementacja");
    for (int p = 0; p < num_presets; p++) {
        printf(" %6s |", presets[p].name);
    }
    printf("\n");

    printf("---------------------|");
    for (int p = 0; p < num_presets; p++) {
        printf("--------|");
    }
    printf("\n");

    for (int i = 0; i < NUM_IMPLEMENTATIONS; i++) {
        printf("%-20s |", impls[i].name);
        for (int p = 0; p < num_presets; p++) {
            printf(" %6.2f |", all_results[p][i]);
        }
        printf("\n");
    }

    printf("================================================================\n");

    // Srednia wydajnosc
    printf("\nSrednia wydajnosc (median across presets):\n");
    for (int i = 0; i < NUM_IMPLEMENTATIONS; i++) {
        double avg = 0.0;
        for (int p = 0; p < num_presets; p++) {
            avg += all_results[p][i];
        }
        avg /= num_presets;
        printf("  %s: %.2f GFLOPS\n", impls[i].name, avg);
    }
    printf("================================================================\n");

    if (json_output_path) {
        write_json_results(impls);
    }
}

static void print_usage(const char* prog) {
    printf("Uzycie:\n");
    printf("  %s                     - uruchamia preset 'medium'\n", prog);
    printf("  %s <preset>            - uruchamia wybrany preset\n", prog);
    printf("  %s all                 - uruchamia wszystkie presety\n", prog);
    printf("  %s <iter> <rows> <cols> - parametry recznie\n", prog);
    printf("\nDostepne presety:\n");
    for (int i = 0; i < num_presets; i++) {
        printf("  %-8s - %dx%d, %d iteracji\n",
               presets[i].name, presets[i].rows, presets[i].cols, presets[i].iterations);
    }
}

int main(int argc, char* argv[]) {
    // Synchronize OpenBLAS thread count with OMP_NUM_THREADS
    int nthreads = omp_get_max_threads();
    openblas_set_num_threads(nthreads);
    goto_set_num_threads(nthreads);

    int has_blis = load_blis(nthreads);

    Implementation impls[NUM_IMPLEMENTATIONS] = {
        {"Naive",           gemv_naive,          {0}, 0, 0, 0, 0, 0},
        {"SIMD",            gemv_simd,           {0}, 0, 0, 0, 0, 0},
        {"SIMD + Prefetch", gemv_simd_prefetch,  {0}, 0, 0, 0, 0, 0},
        {"AVX+FMA Blocked", gemv_avx_fma_blocked,{0}, 0, 0, 0, 0, 0},
        {"AVX+FMA V2",      gemv_avx_fma_v2,    {0}, 0, 0, 0, 0, 0},
        {"AVX+FMA V3",      gemv_avx_fma_v3,    {0}, 0, 0, 0, 0, 0},
        {"AVX+FMA V3_OMP",  gemv_avx_fma_v3_omp,{0}, 0, 0, 0, 0, 0},
        {"OpenBLAS",        openblas_wrapper,    {0}, 0, 0, 0, 0, 0},
        {"BLIS",            blis_wrapper,        {0}, 0, 0, 0, 0, 0},
    };

    // Update library names based on thread count
    char openblas_name[32], blis_name_buf[32];
    snprintf(openblas_name, sizeof(openblas_name), "OpenBLAS (%dT)", nthreads);
    snprintf(blis_name_buf, sizeof(blis_name_buf), "BLIS (%dT)%s", nthreads, has_blis ? "" : " N/A");
    impls[7].name = openblas_name;
    impls[8].name = blis_name_buf;

    /* strip --json <path> from argv before preset dispatch */
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "--json") == 0) {
            json_output_path = argv[i + 1];
            /* remove both args by shifting */
            for (int j = i; j < argc - 2; j++) argv[j] = argv[j + 2];
            argc -= 2;
            break;
        }
    }

    print_header();
    printf("Threads: OMP=%d, OpenBLAS=%d, BLIS=%s\n\n",
           omp_get_max_threads(), openblas_get_num_threads(),
           has_blis ? "loaded" : "not found");

    if (argc == 1) {
        run_single_preset(&presets[2], impls);
    } else if (argc == 2) {
        if (strcmp(argv[1], "all") == 0) {
            run_all_presets(impls);
        } else if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            int found = -1;
            for (int i = 0; i < num_presets; i++) {
                if (strcmp(argv[1], presets[i].name) == 0) {
                    found = i;
                    break;
                }
            }
            if (found >= 0) {
                run_single_preset(&presets[found], impls);
            } else {
                fprintf(stderr, "Nieznany preset: %s\n", argv[1]);
                print_usage(argv[0]);
                return 1;
            }
        }
    } else if (argc == 4) {
        Preset custom = {
            "custom",
            atoi(argv[2]),
            atoi(argv[3]),
            atoi(argv[1])
        };
        run_single_preset(&custom, impls);
    } else {
        print_usage(argv[0]);
        return 1;
    }

    if (blis_handle) dlclose(blis_handle);
    return 0;
}
