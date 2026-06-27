#include "internal.hpp"

#include <abseil/solver.hpp>

#include <array>

namespace abseil {
namespace {

operation_error dispatch_error(operation_code code, std::string message)
{
  return {code, operation_stage::dispatch, 0, std::move(message)};
}

}

operation_result<backend> parse_backend(std::string_view name)
{
  constexpr std::array names{
    std::pair{std::string_view{"native_cpu"}, backend::native_cpu},
    std::pair{std::string_view{"ceres_cpu"}, backend::ceres_cpu},
    std::pair{std::string_view{"native_cuda_cold"}, backend::native_cuda_cold}
  };
  for(const auto& [text, value] : names)
    if(text == name)
      return value;
  return std::unexpected(dispatch_error(operation_code::invalid_argument,
    "backend name is unknown"));
}

std::string_view backend_name(backend value) noexcept
{
  switch(value) {
    case backend::native_cpu: return "native_cpu";
    case backend::ceres_cpu: return "ceres_cpu";
    case backend::native_cuda_cold: return "native_cuda_cold";
  }
  return "unknown";
}

std::string_view termination_name(termination value) noexcept
{
  switch(value) {
    case termination::no_free_nodes: return "no_free_nodes";
    case termination::converged_cost: return "converged_cost";
    case termination::converged_gradient: return "converged_gradient";
    case termination::converged_update: return "converged_update";
    case termination::maximum_iterations: return "maximum_iterations";
    case termination::maximum_time: return "maximum_time";
    case termination::rejection_limit: return "rejection_limit";
    case termination::singular_graph: return "singular_graph";
    case termination::numerical_breakdown: return "numerical_breakdown";
  }
  return "unknown";
}

bool backend_available(backend value) noexcept
{
  switch(value) {
    case backend::native_cpu: return true;
    case backend::ceres_cpu: return detail::ceres_backend_available();
    case backend::native_cuda_cold: return detail::cuda_backend_available();
  }
  return false;
}

std::vector<backend> available_backends()
{
  std::vector<backend> result;
  for(backend value : {backend::native_cpu, backend::ceres_cpu,
      backend::native_cuda_cold})
    if(backend_available(value))
      result.push_back(value);
  return result;
}

operation_result<solve_result> solve(
  const prepared_problem& value,
  const solve_options& options
)
{
  switch(options.selected_backend) {
    case backend::native_cpu:
      if(!std::holds_alternative<native_cpu_options>(options.specific))
        return std::unexpected(dispatch_error(operation_code::invalid_argument,
          "native_cpu backend options do not match"));
      if(!cpu_fields_accessible(value))
        return std::unexpected(dispatch_error(operation_code::invalid_argument,
          "native_cpu cannot access device-resident fields"));
      return detail::solve_native_cpu(value, options.common,
        std::get<native_cpu_options>(options.specific));
    case backend::ceres_cpu:
      if(!std::holds_alternative<ceres_cpu_options>(options.specific))
        return std::unexpected(dispatch_error(operation_code::invalid_argument,
          "ceres_cpu backend options do not match"));
      return detail::solve_ceres_cpu(value, options.common,
        std::get<ceres_cpu_options>(options.specific));
    case backend::native_cuda_cold:
      if(!std::holds_alternative<native_cuda_cold_options>(options.specific))
        return std::unexpected(dispatch_error(operation_code::invalid_argument,
          "native_cuda_cold backend options do not match"));
      return detail::solve_native_cuda_cold(value, options.common,
        std::get<native_cuda_cold_options>(options.specific));
  }
  return std::unexpected(dispatch_error(operation_code::invalid_argument,
    "backend selector is invalid"));
}

}
