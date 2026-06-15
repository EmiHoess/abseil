cmake_minimum_required(VERSION 3.25)

project(seil VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

option(seil_build_benchmarks "Build the seil benchmark executable" OFF)
option(seil_enable_ceres "Build the Ceres CPU backend" OFF)
option(seil_enable_cuda "Build the cold-snapshot CUDA backend" OFF)

add_library(seil INTERFACE)
add_library(seil::seil ALIAS seil)

target_compile_features(seil INTERFACE cxx_std_23)
target_include_directories(seil
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_options(seil
  INTERFACE
    $<$<CXX_COMPILER_ID:MSVC>:/W4;/permissive->
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall;-Wextra;-Wpedantic>
)
