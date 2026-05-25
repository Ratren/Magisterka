cmake_minimum_required(VERSION 3.10)
project(zen3_kernels C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP REQUIRED)

add_subdirectory(common)
add_subdirectory(dot_product)
add_subdirectory(gemv)
add_subdirectory(gemm)
add_subdirectory(conv)
