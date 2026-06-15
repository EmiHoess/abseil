#pragma once

#include <expected>
#include <string>
#include <string_view>

namespace seil {

enum class operation_code {
  invalid_argument,
  invalid_problem,
  backend_unavailable,
  backend_failure,
  resource_exhausted,
  cuda_failure
};

enum class operation_stage {
  validation,
  preparation,
  dispatch,
  device_query,
  context_create,
  upload,
  solve,
  pose_count,
  pose_download,
  iteration_count,
  iteration_download
};

struct operation_error {
  operation_code code{operation_code::invalid_argument};
  operation_stage stage{operation_stage::validation};
  int native_status{};
  std::string message;
};

template<class value_type>
using operation_result = std::expected<value_type, operation_error>;

using operation_status = operation_result<void>;

constexpr std::string_view operation_code_name(operation_code code) noexcept
{
  switch(code) {
    case operation_code::invalid_argument: return "invalid_argument";
    case operation_code::invalid_problem: return "invalid_problem";
    case operation_code::backend_unavailable: return "backend_unavailable";
    case operation_code::backend_failure: return "backend_failure";
    case operation_code::resource_exhausted: return "resource_exhausted";
    case operation_code::cuda_failure: return "cuda_failure";
  }
  return "unknown";
}

constexpr std::string_view operation_stage_name(operation_stage stage) noexcept
{
  switch(stage) {
    case operation_stage::validation: return "validation";
    case operation_stage::preparation: return "preparation";
    case operation_stage::dispatch: return "dispatch";
    case operation_stage::device_query: return "device_query";
    case operation_stage::context_create: return "context_create";
    case operation_stage::upload: return "upload";
    case operation_stage::solve: return "solve";
    case operation_stage::pose_count: return "pose_count";
    case operation_stage::pose_download: return "pose_download";
    case operation_stage::iteration_count: return "iteration_count";
    case operation_stage::iteration_download: return "iteration_download";
  }
  return "unknown";
}

}
