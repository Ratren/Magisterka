cmake_minimum_required(VERSION 3.10)
project(conv_benchmark C)

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

add_library(conv_impl STATIC
    conv_naive.c
    conv_reorder.c
    conv_blocked.c
    conv_microkernel.c
    conv_packed.c
    conv_im2col_gemm.c
    conv_omp.c
    conv_1x1.c
    conv_nchwc.c
    conv_oc_blocked.c
    conv_winograd.c
    conv_libxsmm.c
    conv_zen3.c
)
target_link_libraries(conv_impl PUBLIC OpenMP::OpenMP_C common)

set(OPENBLAS_ROOT ${CMAKE_SOURCE_DIR}/../openblas_install)

target_include_directories(conv_impl PRIVATE ${OPENBLAS_ROOT}/include)

add_executable(conv_benchmark benchmark.c)
target_include_directories(conv_benchmark PRIVATE ${OPENBLAS_ROOT}/include)
target_link_directories(conv_benchmark PRIVATE ${OPENBLAS_ROOT}/lib)
target_link_libraries(conv_benchmark PRIVATE conv_impl common openblas)
set_target_properties(conv_benchmark PROPERTIES
    OUTPUT_NAME benchmark
    BUILD_RPATH ${OPENBLAS_ROOT}/lib
    INSTALL_RPATH ${OPENBLAS_ROOT}/lib)
