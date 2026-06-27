#pragma once

#include "internal.hpp"

#include <span>
#include <vector>

namespace abseil::detail {

struct sparse_system {
  cost_breakdown cost{};
  registration_statistics registration{};
  std::vector<sparse_candidate> blocks;
  std::vector<double> hessian;
  std::vector<double> gradient;
};

operation_result<sparse_system> build_sparse_system(
  const prepared_storage& storage,
  std::span<const pose4> poses
);

}
