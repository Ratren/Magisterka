#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <ctype.h>
#include <omp.h>
#include "gemv.h"
#include "openblas/cblas.h"

extern void openblas_set_num_threads(int num_threads);
extern int openblas_get_num_threads(void);
extern void goto_set_num_threads(int num_threads);

#define NUM_IMPLEMENTATIONS 8
#define WARMUP_ITERATIONS 10

typedef void (*gemv_func)(int, int, double, const double*, const double*, double, double*);

typedef struct {
    const char* name;
    gemv_func func;
    double time_sec;
    double gflops;
    double max_error;
} Implementation;

typedef struct {
    const char* name;
    int rows;
    int cols;
    int iterations;
} Preset;

static Preset presets[] = {
    {"tiny",   64,   64,   500000},
    {"small",  256,  256,  100000},
    {"medium", 1024, 1024, 15000},
    {"large",  4096, 4096, 1500},
    {"wide",   256,  8192, 10000},
    {"tall",   8192, 256,  10000},
};
static const int num_presets = sizeof(presets) / sizeof(presets[0]);

static void openblas_wrapper(int rows, int cols, double alpha,
                             const double* A, const double* x,
                             double beta, double* y) {
    cblas_dgemv(CblasRowMajor, CblasNoTrans, rows, cols, alpha, A, cols, x, 1, beta, y, 1);
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
        
        // Rozgrzewka
        for (int w = 0; w < WARMUP_ITERATIONS; w++) {
            impls[impl].func(rows, cols, alpha, A, x, beta, y);
        }

        // Pomiar (beta=0.0 so implementations handle zeroing y themselves)
        clock_gettime(CLOCK_MONOTONIC, &start);
        for (int i = 0; i < iterations; i++) {
            impls[impl].func(rows, cols, alpha, A, x, beta, y);
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        impls[impl].time_sec = (end.tv_sec - start.tv_sec) + 
                               (end.tv_nsec - start.tv_nsec) / 1e9;
        impls[impl].gflops = (double)(total_flops * iterations) / 
                             (impls[impl].time_sec * 1e9);
        impls[impl].max_error = calc_max_error(y, y_ref, rows);
        
        printf(" %.3fs\n", impls[impl].time_sec);
    }
    
    free(A);
    free(x);
    free(y);
    free(y_ref);
}

static void print_results(Implementation* impls, int rows, int cols, int iterations) {
    double naive_time = impls[0].time_sec;
    double blas_time = impls[NUM_IMPLEMENTATIONS - 1].time_sec;
    
    printf("\n");
    printf("----------------------------------------------------------------\n");
    printf("%-20s | %8s | %6s | %8s | %7s | %s\n", 
           "Implementacja", "Czas (s)", "GFLOPS", "vs Naive", "vs BLAS", "Blad");
    printf("---------------------|----------|--------|----------|---------|----------\n");
    
    for (int i = 0; i < NUM_IMPLEMENTATIONS; i++) {
        printf("%-20s | %8.3f | %6.2f | %7.2fx | %6.2fx | %.1e\n",
               impls[i].name,
               impls[i].time_sec,
               impls[i].gflops,
               naive_time / impls[i].time_sec,
               blas_time / impls[i].time_sec,
               impls[i].max_error);
    }
    printf("----------------------------------------------------------------\n");
    
    int best_idx = 0;
    double best_gflops = 0.0;
    for (int i = 0; i < NUM_IMPLEMENTATIONS - 1; i++) {
        if (impls[i].gflops > best_gflops) {
            best_gflops = impls[i].gflops;
            best_idx = i;
        }
    }
    
    double vs_blas = blas_time / impls[best_idx].time_sec;
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
    
    printf("================================================================\n");
    printf("              GEMV Benchmark - %s\n", cpu_name);
    printf("================================================================\n");
    
    if (l1 > 0 && l2 > 0 && l3 > 0) {
        printf("L1: %d KB | L2: %d KB | L3: %d MB\n", 
               l1 / 1024, l2 / 1024, l3 / (1024 * 1024));
    }
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

static void run_all_presets(Implementation* impls) {
    double all_results[num_presets][NUM_IMPLEMENTATIONS];
    
    for (int p = 0; p < num_presets; p++) {
        printf("\n[ Preset: %s | %dx%d | %d iteracji ]\n\n",
               presets[p].name, presets[p].rows, presets[p].cols, presets[p].iterations);
        printf("Uruchamianie benchmarkow...\n");
        
        run_benchmark(presets[p].rows, presets[p].cols, presets[p].iterations, impls);
        print_results(impls, presets[p].rows, presets[p].cols, presets[p].iterations);
        
        for (int i = 0; i < NUM_IMPLEMENTATIONS; i++) {
            all_results[p][i] = impls[i].gflops;
        }
        
        printf("\n");
    }
    
    // Podsumowanie
    printf("\n================================================================\n");
    printf("                   PODSUMOWANIE (GFLOPS)\n");
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
    printf("\nSrednia wydajnosc:\n");
    for (int i = 0; i < NUM_IMPLEMENTATIONS; i++) {
        double avg = 0.0;
        for (int p = 0; p < num_presets; p++) {
            avg += all_results[p][i];
        }
        avg /= num_presets;
        printf("  %s: %.2f GFLOPS\n", impls[i].name, avg);
    }
    printf("================================================================\n");
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
    // This way OMP_NUM_THREADS=1 makes everything single-threaded,
    // and OMP_NUM_THREADS=N makes both OMP and OpenBLAS use N threads.
    int nthreads = omp_get_max_threads();
    openblas_set_num_threads(nthreads);
    goto_set_num_threads(nthreads);
    
    Implementation impls[NUM_IMPLEMENTATIONS] = {
        {"Naive",           gemv_naive,          0, 0, 0},
        {"SIMD",            gemv_simd,           0, 0, 0},
        {"SIMD + Prefetch", gemv_simd_prefetch,  0, 0, 0},
        {"AVX+FMA Blocked", gemv_avx_fma_blocked, 0, 0, 0},
        {"AVX+FMA V2",      gemv_avx_fma_v2,     0, 0, 0},
        {"AVX+FMA V3",      gemv_avx_fma_v3,     0, 0, 0},
        {"AVX+FMA V3_OMP",  gemv_avx_fma_v3_omp, 0, 0, 0},
        {"OpenBLAS",        openblas_wrapper,    0, 0, 0},
    };
    
    // Update OpenBLAS name based on thread count
    char blas_name[32];
    snprintf(blas_name, sizeof(blas_name), "OpenBLAS (%dT)", nthreads);
    impls[7].name = blas_name;
    
    print_header();
    printf("Threads: OMP=%d, OpenBLAS=%d\n\n", omp_get_max_threads(), openblas_get_num_threads());
    
    if (argc == 1) {
        // Domyslnie: medium
        run_single_preset(&presets[2], impls);
    } else if (argc == 2) {
        if (strcmp(argv[1], "all") == 0) {
            run_all_presets(impls);
        } else if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            // Szukaj presetu
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
    
    return 0;
}
