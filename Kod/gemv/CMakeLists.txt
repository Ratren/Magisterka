cmake_minimum_required(VERSION 3.10)
project(gemv_benchmark C)

set(CMAKE_C_STANDARD 11)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Flagi zoptymalizowane dla AMD Ryzen 5 5600 (Zen 3)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} \
    -O3 \
    -march=znver3 \
    -mtune=znver3 \
    -mavx2 \
    -mfma \
    -funroll-loops \
    -flto \
    -fno-plt")

add_library(gemv_impl STATIC
    gemv_naive.c
    gemv_simd.c
    gemv_simd_prefetch.c
    gemv_avx_fma_blocked.c
    gemv_avx_fma_v2.c
    gemv_avx_fma_v3.c
    gemv_avx_fma_v3_omp.c
)

# Link OpenMP to the library
target_link_libraries(gemv_impl PUBLIC OpenMP::OpenMP_C)

set(OPENBLAS_ROOT ${CMAKE_SOURCE_DIR}/openblas_install)

add_executable(benchmark benchmark.c)
target_include_directories(benchmark PRIVATE ${OPENBLAS_ROOT}/include)
target_link_directories(benchmark PRIVATE ${OPENBLAS_ROOT}/lib)
target_link_libraries(benchmark PRIVATE gemv_impl openblas m dl)
set_target_properties(benchmark PROPERTIES
    BUILD_RPATH ${OPENBLAS_ROOT}/lib
    INSTALL_RPATH ${OPENBLAS_ROOT}/lib)

add_executable(tuning tuning.c)
target_include_directories(tuning PRIVATE ${OPENBLAS_ROOT}/include)
target_link_directories(tuning PRIVATE ${OPENBLAS_ROOT}/lib)
target_link_libraries(tuning PRIVATE openblas m)
set_target_properties(tuning PROPERTIES
    BUILD_RPATH ${OPENBLAS_ROOT}/lib
    INSTALL_RPATH ${OPENBLAS_ROOT}/lib)
