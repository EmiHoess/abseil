cmake_minimum_required(VERSION 3.25)

project(seil VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

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

set(seil_cmake_install_dir "${CMAKE_INSTALL_LIBDIR}/cmake/seil")

install(TARGETS seil
  EXPORT seil_targets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT seil_targets
  FILE seilTargets.cmake
  NAMESPACE seil::
  DESTINATION ${seil_cmake_install_dir}
)

configure_package_config_file(
  cmake/seilConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/seilConfig.cmake
  INSTALL_DESTINATION ${seil_cmake_install_dir}
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/seilConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/seilConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/seilConfigVersion.cmake
  DESTINATION ${seil_cmake_install_dir}
)
