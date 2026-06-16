#ifndef DOT_H
#define DOT_H

#include <stddef.h>

double dot_naive(const double* a, const double* b, size_t n);
double dot_simd(const double* a, const double* b, size_t n);
double dot_simd_multiacc(const double* a, const double* b, size_t n);
double dot_omp(const double* a, const double* b, size_t n);

#endif
