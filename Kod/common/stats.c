#include "common.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

void gen_double_arr(double* arr, size_t n) {
    for (size_t i = 0; i < n; i++)
        arr[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

void gen_float_arr(float* arr, size_t n) {
    for (size_t i = 0; i < n; i++)
        arr[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

double max_abs_diff_d(const double* a, const double* b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

double max_abs_diff_f(const float* a, const float* b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > m) m = d;
    }
    return m;
}

double compute_median(double* vals, int n) {
    double sorted[64];
    if (n > 64) n = 64;
    memcpy(sorted, vals, n * sizeof(double));
    qsort(sorted, n, sizeof(double), cmp_double);
    if (n % 2 == 1) return sorted[n / 2];
    return (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
}

double compute_stddev(const double* vals, int n, double mean) {
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        double d = vals[i] - mean;
        sum_sq += d * d;
    }
    return sqrt(sum_sq / n);
}
