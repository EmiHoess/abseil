#include "sparse_system.hpp"

#include <algorithm>
#include <cmath>

namespace seil::detail {

operation_result<sparse_system> build_sparse_system(
  const prepared_storage& storage,
  std::span<const pose4> poses
)
{
  auto assembled = assemble_objective(storage, poses, true);
  if(!assembled)
    return std::unexpected(std::move(assembled.error()));

  sparse_system output;
  output.cost = assembled->cost;
  output.registration = assembled->registration;
  output.gradient = std::move(assembled->gradient);
  output.blocks = storage.topology.sparse_candidates;
  output.hessian.assign(output.blocks.size() * 16U, 0.0);
  const std::size_t dimension = storage.topology.active_count * 4U;

  for(std::size_t slot = 0; slot < output.blocks.size(); ++slot) {
    const auto& block = output.blocks[slot];
    const std::size_t row_offset = static_cast<std::size_t>(block.row) * 4U;
    const std::size_t column_offset = static_cast<std::size_t>(block.column) * 4U;
    for(std::size_t row = 0; row < 4; ++row)
      for(std::size_t column = 0; column < 4; ++column)
        output.hessian[slot * 16U + row * 4U + column] =
          assembled->dense_hessian[(row_offset + row) * dimension +
            column_offset + column];
  }

  if(!std::all_of(output.gradient.begin(), output.gradient.end(),
      [](double value) { return std::isfinite(value); }) ||
    !std::all_of(output.hessian.begin(), output.hessian.end(),
      [](double value) { return std::isfinite(value); }))
    return std::unexpected(operation_error{operation_code::invalid_problem,
      operation_stage::solve, 0, "normal equations contain non-finite values"});
  return output;
}

}
