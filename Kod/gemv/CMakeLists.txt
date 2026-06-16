cmake_minimum_required(VERSION 3.10)
project(gemv_benchmark C)

set(CMAKE_C_STANDARD 11)

find_package(OpenMP REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} \
    -O3 \
    -march=znver3 \
    -mtune=znver3 \
    -mavx2 \
    -mfma \
    -funroll-loops \
    -flto \
    -fno-plt")

if(NOT TARGET common)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../common ${CMAKE_BINARY_DIR}/common)
endif()

add_library(gemv_impl STATIC
    gemv_naive.c
    gemv_simd.c
    gemv_simd_prefetch.c
    gemv_avx_fma_blocked.c
    gemv_avx_fma_v2.c
    gemv_avx_fma_v3.c
    gemv_avx_fma_v3_omp.c
)
target_link_libraries(gemv_impl PUBLIC OpenMP::OpenMP_C)

set(OPENBLAS_ROOT ${CMAKE_SOURCE_DIR}/../openblas_install)

add_executable(gemv_benchmark benchmark.c)
target_include_directories(gemv_benchmark PRIVATE ${OPENBLAS_ROOT}/include)
target_link_directories(gemv_benchmark PRIVATE ${OPENBLAS_ROOT}/lib)
target_link_libraries(gemv_benchmark PRIVATE gemv_impl common openblas)
set_target_properties(gemv_benchmark PROPERTIES
    OUTPUT_NAME benchmark
    BUILD_RPATH ${OPENBLAS_ROOT}/lib
    INSTALL_RPATH ${OPENBLAS_ROOT}/lib)

add_executable(tuning tuning.c)
target_include_directories(tuning PRIVATE ${OPENBLAS_ROOT}/include)
target_link_directories(tuning PRIVATE ${OPENBLAS_ROOT}/lib)
target_link_libraries(tuning PRIVATE openblas m)
set_target_properties(tuning PROPERTIES
    BUILD_RPATH ${OPENBLAS_ROOT}/lib
    INSTALL_RPATH ${OPENBLAS_ROOT}/lib)
