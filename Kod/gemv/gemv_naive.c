#include "gemv.h"

void gemv_naive(int rows, int cols, double alpha,
                const double* A, const double* x,
                double beta, double* y) {
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < cols; j++) {
            sum += A[i * cols + j] * x[j];
        }
        y[i] = alpha * sum + beta * y[i];
    }
}
