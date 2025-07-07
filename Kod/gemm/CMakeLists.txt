cmake_minimum_required(VERSION 3.10)
project(my_blas_project C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native")

find_package(BLAS REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(main main.c)
target_link_libraries(main PRIVATE OpenMP::OpenMP_C openblas)

add_executable(lvl1 lvl1.c)
add_executable(lvl2 lvl2.c)
add_executable(lvl3 lvl3.c)

target_link_libraries(lvl2 PRIVATE m)
target_link_libraries(lvl3 PRIVATE m)

include_directories(/usr/include)
