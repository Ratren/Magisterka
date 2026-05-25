cmake_minimum_required(VERSION 3.10)
project(dot_product_benchmark C)

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

add_library(dot_impl STATIC
    dot_naive.c
    dot_simd.c
    dot_simd_multiacc.c
    dot_omp.c
)
target_link_libraries(dot_impl PUBLIC OpenMP::OpenMP_C)

set(OPENBLAS_ROOT ${CMAKE_SOURCE_DIR}/../openblas_install)

add_executable(dot_benchmark benchmark.c)
target_include_directories(dot_benchmark PRIVATE ${OPENBLAS_ROOT}/include)
target_link_directories(dot_benchmark PRIVATE ${OPENBLAS_ROOT}/lib)
target_link_libraries(dot_benchmark PRIVATE dot_impl common openblas)
set_target_properties(dot_benchmark PROPERTIES
    OUTPUT_NAME benchmark
    BUILD_RPATH ${OPENBLAS_ROOT}/lib
    INSTALL_RPATH ${OPENBLAS_ROOT}/lib)
