#include "common.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
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
