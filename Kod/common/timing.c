#define _POSIX_C_SOURCE 200809L
#include "common.h"
#include <time.h>

double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}
