#include <bits/time.h>
#include <ctype.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

#define SIZE 1000000000

// 32KB = 32 * 1024 = 32768B - CACHE L1d
// Liczba zmiennych typu double która zmiesci się w cacheu: 32768 / 8 = 4096
// #define BLOCK_SIZE 4032

// 512KB = 512 * 1024 = 524288B - CACHE L2
// Liczba zmiennych typu double która zmiesci się w cacheu: 524288 / 8 = 65536
// #define BLOCK_SIZE 65472 // daję trochę mniej tak aby zmieściły się też inne
// zmienne;

// 32MB = 32 * 1024 * 1024 = 33554432 - CACHE L3
// Liczba zmiennych typu double która zmiesci się w cacheu: 33554432 / 8 =
// 4194304 #define BLOCK_SIZE 41934240

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

  printf("Cache Size: %ld bytes\n", size);
  return size;
}

int dot_product(double* restrict a, double* restrict b, int block_size) {
  double result;
  for (size_t j = 0; j < SIZE; j += block_size) {
    size_t end = (j + block_size > SIZE) ? SIZE : j + block_size;
    for (size_t k = j; k < end; k++) {
      result += a[j] * b[j];
    }
  } 
  return result;
}

int dot_product_simd(double* a, double* b, int block_size) {
  double result;
  double temp[4];
  for (size_t j = 0; j < SIZE; j += block_size) {
    size_t end = (j + block_size > SIZE) ? SIZE : j + block_size;
    __m256d sum_vec = _mm256_setzero_pd();
    size_t i=j;
    for (; i + 3 < end; i+=4) {
      __m256d va = _mm256_loadu_pd(&a[i]);
      __m256d vb = _mm256_loadu_pd(&b[i]);
      __m256d prod = _mm256_mul_pd(va, vb);
      sum_vec = _mm256_add_pd(sum_vec, prod);
    }

    _mm256_storeu_pd(temp, sum_vec);
    result += temp[0] + temp[1] + temp[2] + temp[3];

    for (; i < end; i++) {
      result += a[i] * b[i]; 
    }

  } 

  return result;
}

int main(int argc, char *argv[]) {
  // Przyjmowanie argumentów -------------------------------
  if (argc != 3) {
    printf("Usage: %s <number of iterations> <cache_level>\n", argv[0]);
    return 1;
  }

  int num_of_iterations = atoi(argv[1]);
  int cache_level = atoi(argv[2]);

  int block_size = read_cache_size_bytes(cache_level) / 8 - 64;

  // Generacja danych --------------------------------------
  double *a = aligned_alloc(64U, SIZE * sizeof(double));
  double *b = aligned_alloc(64U, SIZE * sizeof(double));

  srand((unsigned int)time(NULL));

  for (size_t i = 0; i < SIZE; i++) {
    a[i] = (double)rand() / RAND_MAX;
    b[i] = (double)rand() / RAND_MAX;
  }
  // -------------------------------------------------------

  double result;

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  for (int i = 0; i < num_of_iterations; i++) {
    result = dot_product_simd(a, b, block_size);
  }

  clock_gettime(CLOCK_MONOTONIC, &end);

  double elapsed_time =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  printf("Dot product: %f\n", result);
  printf("Elapsed time (seconds): %f\n", elapsed_time);

  free(a);
  free(b);
  return 0;
}
