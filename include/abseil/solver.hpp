#pragma once

#include <abseil/problem.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace abseil {

enum class backend : std::uint32_t {
  native_cpu,
  ceres_cpu,
  native_cuda_cold
};

enum class termination : std::uint32_t {
  no_free_nodes,
  converged_cost,
  converged_gradient,
  converged_update,
  maximum_iterations,
  maximum_time,
  rejection_limit,
  singular_graph,
  numerical_breakdown
};

struct common_options {
  std::size_t maximum_iterations{50};
  double maximum_seconds{30.0};
};

struct native_lm_options {
  std::size_t maximum_rejected_steps{8};
  std::size_t pcg_maximum_iterations{256};
  double initial_damping{1.0e-3};
  double minimum_damping{1.0e-9};
  double maximum_damping{1.0e9};
  double minimum_cost_change{1.0e-8};
  double minimum_gradient_norm{1.0e-8};
  double minimum_update_norm{1.0e-8};
  double maximum_translation_step_m{2.0};
  double maximum_yaw_step_rad{0.5};
  double pcg_tolerance{1.0e-8};
};

struct native_cpu_options {
  native_lm_options lm{};
};

struct ceres_cpu_options {
  double function_tolerance{1.0e-8};
  double gradient_tolerance{1.0e-10};
  double parameter_tolerance{1.0e-8};
  double initial_damping{1.0e-3};
  double minimum_damping{1.0e-9};
  double maximum_damping{1.0e9};
  std::size_t threads{1};
  bool progress_to_stdout{};
};

struct native_cuda_cold_options {
  native_lm_options lm{};
  std::int32_t device_index{};
  std::uint64_t memory_budget_bytes{};
};

using backend_options = std::variant<
  native_cpu_options,
  ceres_cpu_options,
  native_cuda_cold_options
>;

struct solve_options {
  backend selected_backend{backend::native_cpu};
  common_options common{};
  backend_options specific{native_cpu_options{}};
};

struct cost_breakdown {
  double relative{};
  double absolute{};
  double registration{};

  double total() const noexcept
  {
    return relative + absolute + registration;
  }
};

struct registration_statistics {
  std::uint64_t accepted_rows{};
  std::array<std::uint64_t, 9> rejections{};
};

struct iteration_summary {
  std::size_t iteration{};
  std::size_t linear_iterations{};
  bool accepted{};
  double objective_before{};
  double objective_after{};
  double damping{};
  double trust_region_radius{};
  double gradient_norm{};
  double update_norm{};
  double iteration_seconds{};
  double linear_solver_seconds{};
};

struct solver_telemetry {
  double validation_ms{};
  double preparation_ms{};
  double upload_ms{};
  double evaluation_ms{};
  double assembly_ms{};
  double linear_solver_ms{};
  double update_ms{};
  double synchronization_ms{};
  double download_ms{};
  double total_ms{};
  std::uint64_t host_to_device_bytes{};
  std::uint64_t device_to_host_bytes{};
  std::uint64_t peak_device_bytes{};
  std::uint64_t synchronization_count{};
  std::uint64_t accepted_steps{};
  std::uint64_t rejected_steps{};
  std::string implementation;
  std::string device;
};

struct solve_result {
  backend selected_backend{backend::native_cpu};
  termination stop{termination::numerical_breakdown};
  bool solution_usable{};
  std::vector<pose4> poses;
  cost_breakdown initial_cost{};
  cost_breakdown final_cost{};
  registration_statistics registration{};
  std::vector<iteration_summary> iterations;
  solver_telemetry telemetry{};
  std::string message;
};

struct linearization_snapshot {
  cost_breakdown cost{};
  registration_statistics registration{};
  std::vector<std::array<std::int32_t, 2>> sparse_blocks;
  std::vector<double> hessian_blocks;
  std::vector<double> gradient;
};

operation_result<solve_result> solve(
  const prepared_problem& value,
  const solve_options& options
);

operation_result<solve_result> solve_cuda_checked(
  const prepared_problem& value,
  const common_options& common = {},
  const native_cuda_cold_options& options = {}
);

operation_result<backend> parse_backend(std::string_view name);
std::string_view backend_name(backend value) noexcept;
std::string_view termination_name(termination value) noexcept;
bool backend_available(backend value) noexcept;
std::vector<backend> available_backends();

}
