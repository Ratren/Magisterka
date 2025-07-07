#include <ctype.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

int read_cache_size_bytes(int index) {
  char path[128];
  char size_str[32];

  snprintf(path, sizeof(path),
           "/sys/devices/system/cpu/cpu0/cache/index%d/size", index);
  FILE *fp = fopen(path, "r");

  if (fp == NULL) {
    perror("Failed to read cache size");
    return -1;
  }

  if (!fgets(size_str, sizeof(size_str), fp)) {
    fclose(fp);
    return -1;
  }
  fclose(fp);

  size_str[strcspn(size_str, "\n")] = 0;

  long size = strtol(size_str, NULL, 10);
  char unit = size_str[strlen(size_str) - 1];

  switch (toupper(unit)) {
  case 'K':
    size *= 1024;
    break;
  case 'M':
    size *= 1024 * 1024;
    break;
  default:
    break;
  }

  return size;
}

int find_block_size() {
  int cache_size = read_cache_size_bytes(2);
  float usable = cache_size / 3.0;
  int block_size = (int)sqrt(usable/sizeof(float)); //kwadratowe płytki
  // zaokrąglam w dół do wielokrotności 8-ki
  return block_size - (block_size % 8);
}

void gen_float_arr(float* arr, int size) {
  for (int i = 0; i < size; i++) {
    arr[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  }
}

void gemm(int M, int N, int K, float alpha, const float* A, const float* B, float* C, int block_size) {
  memset(C, 0, M * N * sizeof(float));
  for (int ii = 0; ii < M; ii += block_size) {
    for (int jj = 0; jj < N; jj += block_size) {
      for (int kk = 0; kk < K; kk += block_size) {
        int i_max = (ii + block_size> M) ? M : ii + block_size;
        int j_max = (jj + block_size> N) ? N : jj + block_size;
        int k_max = (kk + block_size> K) ? K : kk + block_size;

        for (int i = ii; i < i_max; i++) {
          for (int k = kk; k < k_max; k++) {
            // float a_val = A[i * K + k];
            // for (int j = jj; j < j_max; j++) {
            //   C[i * N + j] += alpha * a_val * B[k * N + j];
            // }
            __m256 a_val = _mm256_set1_ps(alpha * A[i * K + k]);
            int j = jj;
            for (; j+7 < j_max; j += 8) {
              __m256 b_part = _mm256_loadu_ps(&B[k * N + j]);
              __m256 c_part = _mm256_loadu_ps(&B[k * N + j]);

              // __m256 mult = _mm256_mul_ps(a_val, b_part);
              // c_part = _mm256_add_ps(c_part, mult);

              // _mm256_storeu_ps(&C[i * N + j], c_part);
              
              __m256 result = _mm256_fmadd_ps(a_val, b_part, c_part);
              _mm256_storeu_ps(&C[i * N + j], result);
            }

            for (; j < j_max; j++) {
              C[i * N + j] += alpha * A[i + K + k] * B[k * N + j];
            }
          }
        }
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

  int block_size = find_block_size();

  for (int i = 0; i < num_of_iterations; i++) {
    gemm(M, N, K, alpha, A, B, C, block_size);
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
