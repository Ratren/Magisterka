#include <time.h>
#include "stdlib.h"
#include "time.h"
#include <stdio.h>
#include <stddef.h>

void gen_float_arr(float* arr, int size) {
  for (int i = 0; i < size; i++) {
    arr[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  }
}

void gemv(int n_rows, int n_cols, float alpha, float* A, float* x, float beta, float* y) {
  for (int i = 0; i < n_rows; i++) {
    float dot = 0.0f;
    for (int j = 0; j < n_cols; j++) {
      dot += A[i * n_cols + j] * x[j];
    }
    y[i] = alpha * dot + beta * y[i];
  }
}

int main(int argc, char *argv[]) {
// Przyjmowanie argumentów --------------------------------------
  if (argc != 4) {
    printf("Usage: %s <number of iterations> <n_rows> <n_cols>\n", argv[0]);
    return 1;
  }

  int num_of_iterations = atoi(argv[1]);
  int n_rows = atoi(argv[2]);
  int n_cols = atoi(argv[3]);
// Generacja danych ---------------------------------------------
  
  float* A = (float*)malloc(n_rows * n_cols * sizeof(float)); // Macierz
  float* x = (float*)malloc(n_cols * sizeof(float)); // Przez ten wektor mnożymy macierz
  float* y = (float*)malloc(n_rows * sizeof(float)); // Wektor wynikowy

  //Funkcja cblas_sgemv() wykonuje działanie Y <- alpha * A * X + beta * Y dlatego:
  float alpha = 1.0f;
  float beta = 0.0f;

  srand(time(NULL));
  gen_float_arr(A, n_rows*n_cols);
  gen_float_arr(x, n_cols);

// --------------------------------------------------------------
  
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start); 

  for (int i=0; i<num_of_iterations; i++) {
    gemv(n_rows, n_cols, alpha, A, x, beta, y);
  }
  
  clock_gettime(CLOCK_MONOTONIC, &end);
  double elapsed_time = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / 1e9;

  printf("Elapsed time (seconds): %f\n", elapsed_time);
  if (n_rows <= 20) {
    printf("Result vector y:\n");
    for (int i = 0; i < n_rows; i++) {
        printf("%f\n", y[i]);
    } 
  }
}
