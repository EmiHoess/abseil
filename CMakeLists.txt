cmake_minimum_required(VERSION 3.25)

project(abseil VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

option(abseil_build_benchmarks "Build the abseil benchmark executable" OFF)
option(abseil_enable_ceres "Build the Ceres CPU backend" OFF)
option(abseil_enable_cuda "Build the cold-snapshot CUDA backend" OFF)

if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
  if(abseil_enable_ceres)
    add_compile_definitions(_ITERATOR_DEBUG_LEVEL=0)
  endif()
endif()

add_library(abseil STATIC
  src/backend_dispatch.cpp
  src/field.cpp
  src/field_views.cpp
  src/geometry.cpp
  src/linear_solver.cpp
  src/native_solver.cpp
  src/objective.cpp
  src/problem.cpp
  src/sparse_system.cpp
)
add_library(abseil::abseil ALIAS abseil)

if(MSVC AND abseil_enable_ceres)
  target_compile_definitions(abseil PUBLIC _ITERATOR_DEBUG_LEVEL=0)
endif()

if(abseil_enable_ceres)
  find_package(Ceres CONFIG REQUIRED)
  target_sources(abseil PRIVATE src/ceres_backend.cpp)
  target_link_libraries(abseil PUBLIC Ceres::ceres)
  target_compile_definitions(abseil PRIVATE ABSEIL_HAS_CERES=1)
else()
  target_sources(abseil PRIVATE src/ceres_stub.cpp)
endif()

if(abseil_enable_cuda)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(abseil PRIVATE
    src/cuda_backend.cpp
    src/cuda_checked.cpp
    src/native_cuda_cold/solver.cu
  )
  target_link_libraries(abseil PUBLIC CUDA::cudart)
  target_compile_definitions(abseil PRIVATE ABSEIL_HAS_CUDA=1)
  set_target_properties(abseil PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
else()
  target_sources(abseil PRIVATE src/cuda_stub.cpp)
endif()

target_compile_features(abseil PUBLIC cxx_std_23)
target_include_directories(abseil
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_options(abseil
  PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4;/permissive->
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

if(abseil_build_benchmarks)
  add_executable(abseil_benchmark apps/benchmark.cpp)
  target_link_libraries(abseil_benchmark PRIVATE abseil::abseil)
endif()

if(BUILD_TESTING)
  add_executable(abseil_error_tests tests/error_tests.cpp)
  target_link_libraries(abseil_error_tests PRIVATE abseil::abseil)
  add_test(NAME abseil_error_tests COMMAND abseil_error_tests)

  add_executable(abseil_geometry_tests tests/geometry_tests.cpp)
  target_link_libraries(abseil_geometry_tests PRIVATE abseil::abseil)
  add_test(NAME abseil_geometry_tests COMMAND abseil_geometry_tests)

  add_executable(abseil_problem_tests tests/problem_tests.cpp)
  target_link_libraries(abseil_problem_tests PRIVATE abseil::abseil)
  add_test(NAME abseil_problem_tests COMMAND abseil_problem_tests)

  add_executable(abseil_field_tests tests/field_tests.cpp)
  target_link_libraries(abseil_field_tests PRIVATE abseil::abseil)
  add_test(NAME abseil_field_tests COMMAND abseil_field_tests)

  add_executable(abseil_objective_tests tests/objective_tests.cpp)
  target_link_libraries(abseil_objective_tests PRIVATE abseil::abseil)
  add_test(NAME abseil_objective_tests COMMAND abseil_objective_tests)

  add_executable(abseil_sparse_system_tests tests/sparse_system_tests.cpp)
  target_link_libraries(abseil_sparse_system_tests PRIVATE abseil::abseil)
  add_test(NAME abseil_sparse_system_tests COMMAND abseil_sparse_system_tests)

  add_executable(abseil_linear_solver_tests tests/linear_solver_tests.cpp)
  target_link_libraries(abseil_linear_solver_tests PRIVATE abseil::abseil)
  add_test(NAME abseil_linear_solver_tests COMMAND abseil_linear_solver_tests)

  add_executable(abseil_native_solver_tests tests/native_solver_tests.cpp)
  target_link_libraries(abseil_native_solver_tests PRIVATE abseil::abseil)
  add_test(NAME abseil_native_solver_tests COMMAND abseil_native_solver_tests)

  add_executable(abseil_backend_dispatch_tests tests/backend_dispatch_tests.cpp)
  target_link_libraries(abseil_backend_dispatch_tests PRIVATE abseil::abseil)
  add_test(NAME abseil_backend_dispatch_tests COMMAND abseil_backend_dispatch_tests)

  add_executable(abseil_backend_parity_tests tests/backend_parity_tests.cpp)
  target_link_libraries(abseil_backend_parity_tests PRIVATE abseil::abseil)
  add_test(NAME abseil_backend_parity_tests COMMAND abseil_backend_parity_tests)

  add_executable(abseil_cuda_adapter_tests tests/cuda_adapter_tests.cpp)
  target_link_libraries(abseil_cuda_adapter_tests PRIVATE abseil::abseil)
  if(abseil_enable_cuda)
    target_compile_definitions(abseil_cuda_adapter_tests PRIVATE ABSEIL_HAS_CUDA=1)
  endif()
  add_test(NAME abseil_cuda_adapter_tests COMMAND abseil_cuda_adapter_tests)

  add_executable(abseil_field_view_tests tests/field_view_tests.cpp)
  target_link_libraries(abseil_field_view_tests PRIVATE abseil::abseil)
  if(abseil_enable_cuda)
    target_compile_definitions(abseil_field_view_tests PRIVATE ABSEIL_HAS_CUDA=1)
  endif()
  add_test(NAME abseil_field_view_tests COMMAND abseil_field_view_tests)

  if(abseil_build_benchmarks)
    add_executable(abseil_benchmark_cli_tests tests/benchmark_cli_tests.cpp)
    add_dependencies(abseil_benchmark_cli_tests abseil_benchmark)
    add_test(NAME abseil_benchmark_cli_tests
      COMMAND abseil_benchmark_cli_tests $<TARGET_FILE:abseil_benchmark>)
  endif()
endif()

set(abseil_cmake_install_dir "${CMAKE_INSTALL_LIBDIR}/cmake/abseil")

install(TARGETS abseil
  EXPORT abseil_targets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT abseil_targets
  FILE abseilTargets.cmake
  NAMESPACE abseil::
  DESTINATION ${abseil_cmake_install_dir}
)

configure_package_config_file(
  cmake/abseilConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/abseilConfig.cmake
  INSTALL_DESTINATION ${abseil_cmake_install_dir}
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/abseilConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/abseilConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/abseilConfigVersion.cmake
  DESTINATION ${abseil_cmake_install_dir}
)
