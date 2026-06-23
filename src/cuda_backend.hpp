#pragma once

#include "native_cuda_cold/cuda_abi.h"

#include <cstdint>
#include <string>

namespace seil::detail {

enum class cuda_stage_status : std::int32_t {
  success,
  unavailable,
  invalid_abi,
  invalid_argument,
  resource_exhausted,
  runtime_failure,
  numerical_failure
};

struct cuda_call_result {
  cuda_stage_status status{cuda_stage_status::success};
  std::int32_t native_status{};
  std::string message;

  explicit operator bool() const noexcept
  {
    return status == cuda_stage_status::success;
  }
};

constexpr std::uint32_t cuda_abi_version = seil_cuda_abi_version_value;

}
