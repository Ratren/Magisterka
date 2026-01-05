cmake_minimum_required(VERSION 3.10)
project(gemv_benchmark C)

set(CMAKE_C_STANDARD 11)

# Flagi zoptymalizowane dla AMD Ryzen 5 5600 (Zen 3)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} \
    -O3 \
    -march=znver3 \
    -mtune=znver3 \
    -mavx2 \
    -mfma \
    -funroll-loops \
    -ffast-math \
    -flto \
    -fno-plt")

add_library(gemv_impl STATIC
    gemv_naive.c
    gemv_simd.c
    gemv_simd_prefetch.c
    gemv_avx_fma_blocked.c
    gemv_avx_fma_v2.c
    gemv_avx_fma_v3.c
)

add_executable(benchmark benchmark.c)
target_link_libraries(benchmark PRIVATE gemv_impl openblas m)

add_executable(tuning tuning.c)
target_link_libraries(tuning PRIVATE openblas m)

include_directories(/usr/include)
