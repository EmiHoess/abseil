cmake_minimum_required(VERSION 3.25)

project(seil VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

include(CTest)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

option(seil_build_benchmarks "Build the seil benchmark executable" OFF)
option(seil_enable_ceres "Build the Ceres CPU backend" OFF)
option(seil_enable_cuda "Build the cold-snapshot CUDA backend" OFF)

add_library(seil STATIC
  src/field.cpp
  src/geometry.cpp
  src/linear_solver.cpp
  src/native_solver.cpp
  src/objective.cpp
  src/problem.cpp
  src/sparse_system.cpp
)
add_library(seil::seil ALIAS seil)

target_compile_features(seil PUBLIC cxx_std_23)
target_include_directories(seil
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_options(seil
  PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4;/permissive->
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall;-Wextra;-Wpedantic>
)

if(BUILD_TESTING)
  add_executable(seil_error_tests tests/error_tests.cpp)
  target_link_libraries(seil_error_tests PRIVATE seil::seil)
  add_test(NAME seil_error_tests COMMAND seil_error_tests)

  add_executable(seil_geometry_tests tests/geometry_tests.cpp)
  target_link_libraries(seil_geometry_tests PRIVATE seil::seil)
  add_test(NAME seil_geometry_tests COMMAND seil_geometry_tests)

  add_executable(seil_problem_tests tests/problem_tests.cpp)
  target_link_libraries(seil_problem_tests PRIVATE seil::seil)
  add_test(NAME seil_problem_tests COMMAND seil_problem_tests)

  add_executable(seil_field_tests tests/field_tests.cpp)
  target_link_libraries(seil_field_tests PRIVATE seil::seil)
  add_test(NAME seil_field_tests COMMAND seil_field_tests)

  add_executable(seil_objective_tests tests/objective_tests.cpp)
  target_link_libraries(seil_objective_tests PRIVATE seil::seil)
  add_test(NAME seil_objective_tests COMMAND seil_objective_tests)

  add_executable(seil_sparse_system_tests tests/sparse_system_tests.cpp)
  target_link_libraries(seil_sparse_system_tests PRIVATE seil::seil)
  add_test(NAME seil_sparse_system_tests COMMAND seil_sparse_system_tests)

  add_executable(seil_linear_solver_tests tests/linear_solver_tests.cpp)
  target_link_libraries(seil_linear_solver_tests PRIVATE seil::seil)
  add_test(NAME seil_linear_solver_tests COMMAND seil_linear_solver_tests)

  add_executable(seil_native_solver_tests tests/native_solver_tests.cpp)
  target_link_libraries(seil_native_solver_tests PRIVATE seil::seil)
  add_test(NAME seil_native_solver_tests COMMAND seil_native_solver_tests)
endif()

set(seil_cmake_install_dir "${CMAKE_INSTALL_LIBDIR}/cmake/seil")

install(TARGETS seil
  EXPORT seil_targets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

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
