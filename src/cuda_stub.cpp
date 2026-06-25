#include "internal.hpp"
#include "cuda_backend.hpp"

namespace {

const char* unavailable_message = "native_cuda_cold backend is not available in this build";

}

extern "C" int seil_cuda_abi_version(void)
{
  return seil_cuda_abi_version_value;
}

extern "C" const char* seil_cuda_last_error(void)
{
  return unavailable_message;
}

extern "C" int seil_cuda_device_count(int32_t* output_count)
{
  if(output_count != nullptr)
    *output_count = 0;
  return static_cast<int>(seil::detail::cuda_stage_status::unavailable);
}

extern "C" int seil_cuda_create(
  const seil_cuda_options*,
  seil_cuda_context** output_context
)
{
  if(output_context != nullptr)
    *output_context = nullptr;
  return static_cast<int>(seil::detail::cuda_stage_status::unavailable);
}

extern "C" int seil_cuda_upload_problem(
  seil_cuda_context*,
  const seil_cuda_problem*
)
{
  return static_cast<int>(seil::detail::cuda_stage_status::unavailable);
}

extern "C" int seil_cuda_solve_stage(seil_cuda_context*, seil_cuda_summary*)
{
  return static_cast<int>(seil::detail::cuda_stage_status::unavailable);
}

extern "C" int seil_cuda_download_poses(
  seil_cuda_context*,
  seil_cuda_pose4*,
  size_t,
  size_t* output_count
)
{
  if(output_count != nullptr)
    *output_count = 0;
  return static_cast<int>(seil::detail::cuda_stage_status::unavailable);
}

extern "C" int seil_cuda_download_iterations(
  seil_cuda_context*,
  seil_cuda_iteration*,
  size_t,
  size_t* output_count
)
{
  if(output_count != nullptr)
    *output_count = 0;
  return static_cast<int>(seil::detail::cuda_stage_status::unavailable);
}

extern "C" int seil_cuda_destroy(seil_cuda_context*)
{
  return static_cast<int>(seil::detail::cuda_stage_status::success);
}

namespace seil::detail {

bool cuda_backend_available() noexcept
{
  return false;
}

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

namespace seil {

operation_result<solve_result> solve_cuda_checked(
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
