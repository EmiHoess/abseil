#pragma once

#include "native_cuda_cold/cuda_abi.h"

#include <seil/problem.hpp>

#include <cstdint>
#include <string>
#include <vector>

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

struct packed_cuda_problem {
  std::vector<seil_cuda_node> nodes;
  std::vector<seil_cuda_edge> relative_edges;
  std::vector<seil_cuda_edge> absolute_edges;
  std::vector<seil_cuda_registration_group> registration_groups;
  std::vector<seil_cuda_registration_sample> registration_samples;
  std::vector<seil_cuda_volume> volumes;
  std::vector<seil_cuda_tsdf_tile> tsdf_tiles;
  std::vector<seil_cuda_esdf_tile> esdf_tiles;
  std::vector<seil_cuda_sparse_slot> sparse_slots;
  seil_cuda_problem view{};
  std::uint64_t byte_count{};
};

operation_result<packed_cuda_problem> pack_cuda_problem(
  const prepared_problem& value
);

}
