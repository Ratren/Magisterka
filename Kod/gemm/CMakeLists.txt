cmake_minimum_required(VERSION 3.10)
project(gemm_benchmark C)

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

add_library(gemm_impl STATIC
    gemm_naive.c
    gemm_ikj.c
    gemm_blocked.c
    gemm_microkernel.c
    gemm_packed.c
    gemm_omp.c
    gemm_zen3.c
    gemm_zen3_intrin.c
    gemm_zen3_sched.c
    gemm_zen3_par.c
    gemm_zen3_tuned.c
    gemm_zen3_tiny.c
    gemm_zen3_best.c
    gemm_strassen.c
)
target_link_libraries(gemm_impl PUBLIC OpenMP::OpenMP_C)

set(OPENBLAS_ROOT ${CMAKE_SOURCE_DIR}/../openblas_install)

add_executable(gemm_benchmark benchmark.c)
target_include_directories(gemm_benchmark PRIVATE ${OPENBLAS_ROOT}/include)
target_link_directories(gemm_benchmark PRIVATE ${OPENBLAS_ROOT}/lib)
target_link_libraries(gemm_benchmark PRIVATE gemm_impl common openblas)
set_target_properties(gemm_benchmark PROPERTIES
    OUTPUT_NAME benchmark
    BUILD_RPATH ${OPENBLAS_ROOT}/lib
    INSTALL_RPATH ${OPENBLAS_ROOT}/lib)
