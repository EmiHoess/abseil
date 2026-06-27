#include "internal.hpp"

namespace abseil::detail {

bool ceres_backend_available() noexcept
{
  return false;
}

operation_result<solve_result> solve_ceres_cpu(
  const prepared_problem&,
  const common_options&,
  const ceres_cpu_options&
)
{
  return std::unexpected(operation_error{
    operation_code::backend_unavailable,
    operation_stage::dispatch,
    0,
    "ceres_cpu backend is not available in this build"
  });
}

}
