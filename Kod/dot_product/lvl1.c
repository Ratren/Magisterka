#include <bits/time.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define SIZE 1000000000

int main(int argc, char *argv[])
{
// Przyjmowanie argumentów -------------------------------
  if (argc != 2) {
    printf("Usage: %s <number of iterations>\n", argv[0]);
    return 1;
  }

  int num_of_iterations = atoi(argv[1]);

// Generacja danych --------------------------------------
  double *a = malloc(SIZE * sizeof(double));
  double *b = malloc(SIZE * sizeof(double));

  srand((unsigned int)time(NULL));

  for (size_t i=0; i < SIZE; i++) {
    a[i] = (double)rand() / RAND_MAX;
    b[i] = (double)rand() / RAND_MAX;
  }
// -------------------------------------------------------
  
  double result; 

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  for (int i=0; i<num_of_iterations; i++) {
    for (size_t j = 0; j < SIZE; j++) {
       result += a[j] * b[j];
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &end);

  double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  printf("Dot product: %f\n", result);
  printf("Elapsed time (seconds): %f\n", elapsed_time);


  free(a);
  free(b);
  return 0;
}

