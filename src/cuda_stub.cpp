#include "internal.hpp"
#include "cuda_backend.hpp"

namespace {

const char* unavailable_message = "native_cuda_cold backend is not available in this build";

}

extern "C" int abseil_cuda_abi_version(void)
{
  return abseil_cuda_abi_version_value;
}

extern "C" const char* abseil_cuda_last_error(void)
{
  return unavailable_message;
}

extern "C" int abseil_cuda_device_count(int32_t* output_count)
{
  if(output_count != nullptr)
    *output_count = 0;
  return static_cast<int>(abseil::detail::cuda_stage_status::unavailable);
}

extern "C" int abseil_cuda_create(
  const abseil_cuda_options*,
  abseil_cuda_context** output_context
)
{
  if(output_context != nullptr)
    *output_context = nullptr;
  return static_cast<int>(abseil::detail::cuda_stage_status::unavailable);
}

extern "C" int abseil_cuda_upload_problem(
  abseil_cuda_context*,
  const abseil_cuda_problem*
)
{
  return static_cast<int>(abseil::detail::cuda_stage_status::unavailable);
}

extern "C" int abseil_cuda_solve_stage(abseil_cuda_context*, abseil_cuda_summary*)
{
  return static_cast<int>(abseil::detail::cuda_stage_status::unavailable);
}

extern "C" int abseil_cuda_download_poses(
  abseil_cuda_context*,
  abseil_cuda_pose4*,
  size_t,
  size_t* output_count
)
{
  if(output_count != nullptr)
    *output_count = 0;
  return static_cast<int>(abseil::detail::cuda_stage_status::unavailable);
}

extern "C" int abseil_cuda_download_iterations(
  abseil_cuda_context*,
  abseil_cuda_iteration*,
  size_t,
  size_t* output_count
)
{
  if(output_count != nullptr)
    *output_count = 0;
  return static_cast<int>(abseil::detail::cuda_stage_status::unavailable);
}

extern "C" int abseil_cuda_destroy(abseil_cuda_context*)
{
  return static_cast<int>(abseil::detail::cuda_stage_status::success);
}

namespace abseil::detail {

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

namespace abseil {

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
