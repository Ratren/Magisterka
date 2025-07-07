#include <ctype.h>
#include <string.h>
#include <time.h>
#include "stdlib.h"
#include "time.h"
#include <stdio.h>
#include <stddef.h>
#include <immintrin.h>
#include <xmmintrin.h>

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


inline int dot_product_simd(float* a, float* b, size_t arr_size, int block_size) {
  float result = 0.0f;
  float temp[8];
  for (size_t j = 0; j < arr_size; j += block_size) {
    size_t end = (j + block_size > arr_size) ? arr_size : j + block_size;
    __m256 sum_vec = _mm256_setzero_ps();
    size_t i=j;
    for (; i + 7 < end; i+=8) {
      _mm_prefetch((const char*)&a[i+16], _MM_HINT_T0);
      _mm_prefetch((const char*)&b[i+16], _MM_HINT_T0);
      __m256 va = _mm256_load_ps(&a[i]);
      __m256 vb = _mm256_load_ps(&b[i]);
      __m256 prod = _mm256_mul_ps(va, vb);
      sum_vec = _mm256_add_ps(sum_vec, prod);
    }
    
    _mm256_store_ps(temp, sum_vec);
    result += temp[0] + temp[1] + temp[2] + temp[3] + 
              temp[4] + temp[5] + temp[6] + temp[7];

    for (; i < end; i++) {
      result += a[i] * b[i]; 
    }

  } 

  return result;
}

void gen_float_arr(float* arr, int size) {
  for (int i = 0; i < size; i++) {
    arr[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  }
}

void gemv(int n_rows, int n_cols, float alpha, float* A, float* x, float beta, float* y, int block_size) {
  for (int i = 0; i < n_rows; i++) {
    float dot = dot_product_simd(A + i*n_cols, x, n_cols, block_size);
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
  
  float* A = aligned_alloc(32, n_rows * n_cols * sizeof(float)); // Macierz
  float* x = aligned_alloc(32, n_cols * sizeof(float)); // Przez ten wektor mnożymy macierz
  float* y = aligned_alloc(32, n_rows * sizeof(float)); // Wektor wynikowy

  //Funkcja cblas_sgemv() wykonuje działanie Y <- alpha * A * X + beta * Y dlatego:
  float alpha = 1.0f;
  float beta = 0.0f;

  srand(time(NULL));
  gen_float_arr(A, n_rows*n_cols);
  gen_float_arr(x, n_cols);

// --------------------------------------------------------------
  
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start); 

  int block_size = read_cache_size_bytes(2) / 4;
  for (int i=0; i<num_of_iterations; i++) {
    gemv(n_rows, n_cols, alpha, A, x, beta, y, block_size);
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
