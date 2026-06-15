cmake_minimum_required(VERSION 3.10)

if(NOT TARGET OpenMP::OpenMP_C)
    find_package(OpenMP REQUIRED)
endif()

add_library(common STATIC
    timing.c
    stats.c
    sysinfo.c
    blis_loader.c
    preset.c
    json_out.c
    runner.c
)

target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(common PUBLIC OpenMP::OpenMP_C m dl)
target_compile_options(common PUBLIC
    -O3 -march=znver3 -mtune=znver3 -mavx2 -mfma
    -funroll-loops -flto -fno-plt)
