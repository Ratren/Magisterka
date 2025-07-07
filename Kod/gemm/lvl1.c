#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void gen_float_arr(float* arr, int size) {
  for (int i = 0; i < size; i++) {
    arr[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  }
}

void naive_gemm(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
  memset(C, 0, M * N * sizeof(float));

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        C[i * N + j] += alpha * A[i * K + k] * B[k * N + j];
      }
    }
  }
}

int main(int argc, char *argv[]) {
  // Przyjmowanie argumentów -------------------------------------------
  if (argc != 5) {
    printf("Usage: %s <number of iterations> <M> <K> <N>\n", argv[0]);
    return 1;
  }

  int num_of_iterations = atoi(argv[1]);
  int M = atoi(argv[2]);
  int K = atoi(argv[3]);
  int N = atoi(argv[4]);

  // Generacja danych --------------------------------------------------
  float* A = (float*)malloc(M * K * sizeof(float));
  float* B = (float*)malloc(K * N * sizeof(float));
  float* C = (float*)malloc(M * N * sizeof(float));

  float alpha = 1.0f;
  float beta = 0.0f;

  srand(time(NULL));
  gen_float_arr(A, M * K);
  gen_float_arr(B, K * N);
  // --------------------------------------------------------------------


  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  for (int i = 0; i < num_of_iterations; i++) {
    naive_gemm(M, N, K, alpha, A, B, beta, C);
  }

  clock_gettime(CLOCK_MONOTONIC, &end);
  double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  printf("Elapsed time (seconds): %f\n", elapsed_time);

  if (M <= 20 && N <= 20) {
    printf("Result matrix C:\n");
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        printf("%f ", C[i * N + j]);
      }
      printf("\n");
    }
  }

  // Free memory
  free(A);
  free(B);
  free(C);

  return 0;
}
