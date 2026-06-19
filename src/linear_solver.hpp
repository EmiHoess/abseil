#pragma once

#include "sparse_system.hpp"

#include <cstddef>
#include <vector>

namespace seil::detail {

struct linear_solution {
  termination stop{termination::numerical_breakdown};
  bool converged{};
  std::size_t iterations{};
  std::vector<double> update;
};

linear_solution solve_sparse_system(
  const topology_index& topology,
  const sparse_system& system,
  double damping,
  const native_lm_options& options
);

}
