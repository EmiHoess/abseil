#include "internal.hpp"

namespace seil::detail {

operation_result<solve_result> solve_native_cuda_cold(
  const prepared_problem&,
  const common_options&,
  const native_cuda_cold_options&
)
{
  return std::unexpected(operation_error{
    operation_code::backend_unavailable,
    operation_stage::dispatch,
    0,
    "native_cuda_cold backend is not available in this build"
  });
}

}
