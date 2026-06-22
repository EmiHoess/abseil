#include "linear_solver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <span>

namespace seil::detail {
namespace {

double dot(std::span<const double> left, std::span<const double> right)
{
  double result = 0.0;
  for(std::size_t index = 0; index < left.size(); ++index)
    result += left[index] * right[index];
  return result;
}

bool invert_block(std::array<double, 16> matrix, std::array<double, 16>& inverse)
{
  inverse = identity_matrix4();
  for(std::size_t column = 0; column < 4; ++column) {
    std::size_t pivot = column;
    for(std::size_t row = column + 1; row < 4; ++row)
      if(std::abs(matrix[row * 4U + column]) >
        std::abs(matrix[pivot * 4U + column]))
        pivot = row;
    if(!(std::abs(matrix[pivot * 4U + column]) > 1.0e-15))
      return false;
    if(pivot != column)
      for(std::size_t item = 0; item < 4; ++item) {
        std::swap(matrix[column * 4U + item], matrix[pivot * 4U + item]);
        std::swap(inverse[column * 4U + item], inverse[pivot * 4U + item]);
      }
    const double scale = matrix[column * 4U + column];
    for(std::size_t item = 0; item < 4; ++item) {
      matrix[column * 4U + item] /= scale;
      inverse[column * 4U + item] /= scale;
    }
    for(std::size_t row = 0; row < 4; ++row) {
      if(row == column)
        continue;
      const double factor = matrix[row * 4U + column];
      for(std::size_t item = 0; item < 4; ++item) {
        matrix[row * 4U + item] -= factor * matrix[column * 4U + item];
        inverse[row * 4U + item] -= factor * inverse[column * 4U + item];
      }
    }
  }
  return true;
}

void multiply(
  const sparse_system& system,
  std::span<const double> input,
  double damping,
  std::span<double> output
)
{
  for(std::size_t index = 0; index < input.size(); ++index)
    output[index] = damping * input[index];
  for(std::size_t slot = 0; slot < system.blocks.size(); ++slot) {
    const auto& coordinate = system.blocks[slot];
    const double* block = system.hessian.data() + slot * 16U;
    const std::size_t row_offset = static_cast<std::size_t>(coordinate.row) * 4U;
    const std::size_t column_offset = static_cast<std::size_t>(coordinate.column) * 4U;
    for(std::size_t row = 0; row < 4; ++row)
      for(std::size_t column = 0; column < 4; ++column) {
        output[row_offset + row] += block[row * 4U + column] *
          input[column_offset + column];
        if(coordinate.row != coordinate.column)
          output[column_offset + column] += block[row * 4U + column] *
            input[row_offset + row];
      }
  }
}

}

linear_solution solve_sparse_system(
  const topology_index& topology,
  const sparse_system& system,
  double damping,
  const native_lm_options& options,
  linear_workspace& workspace
)
{
  linear_solution result;
  if(topology.unanchored_components != 0) {
    result.stop = termination::singular_graph;
    return result;
  }
  const std::size_t dimension = system.gradient.size();
  result.update.assign(dimension, 0.0);
  if(dimension == 0) {
    result.stop = termination::no_free_nodes;
    result.converged = true;
    return result;
  }

  workspace.inverse_blocks.resize(topology.active_count);
  for(std::size_t block_index = 0; block_index < topology.active_count; ++block_index) {
    const auto slot = sparse_slot(topology, static_cast<std::int32_t>(block_index),
      static_cast<std::int32_t>(block_index));
    if(slot < 0) {
      result.stop = termination::singular_graph;
      return result;
    }
    std::array<double, 16> diagonal{};
    std::copy_n(system.hessian.data() + static_cast<std::size_t>(slot) * 16U,
      16, diagonal.data());
    for(std::size_t axis = 0; axis < 4; ++axis)
      diagonal[axis * 4U + axis] += damping;
    if(!invert_block(diagonal, workspace.inverse_blocks[block_index])) {
      result.stop = termination::singular_graph;
      return result;
    }
  }

  workspace.residual.resize(dimension);
  workspace.preconditioned.resize(dimension);
  workspace.direction.resize(dimension);
  workspace.product.resize(dimension);
  for(std::size_t index = 0; index < dimension; ++index)
    workspace.residual[index] = -system.gradient[index];
  const auto apply_preconditioner = [&] {
    for(std::size_t block = 0; block < topology.active_count; ++block)
      for(std::size_t row = 0; row < 4; ++row) {
        double value = 0.0;
        for(std::size_t column = 0; column < 4; ++column)
          value += workspace.inverse_blocks[block][row * 4U + column] *
            workspace.residual[block * 4U + column];
        workspace.preconditioned[block * 4U + row] = value;
      }
  };
  apply_preconditioner();
  workspace.direction = workspace.preconditioned;
  double residual_projection = dot(workspace.residual, workspace.preconditioned);
  const double initial_norm = std::sqrt(std::max(0.0,
    dot(workspace.residual, workspace.residual)));
  if(initial_norm <= options.pcg_tolerance) {
    result.stop = termination::converged_gradient;
    result.converged = true;
    return result;
  }

  for(std::size_t iteration = 0; iteration < options.pcg_maximum_iterations; ++iteration) {
    multiply(system, workspace.direction, damping, workspace.product);
    const double denominator = dot(workspace.direction, workspace.product);
    if(!(denominator > 0.0) || !std::isfinite(denominator)) {
      result.stop = termination::numerical_breakdown;
      return result;
    }
    const double alpha = residual_projection / denominator;
    for(std::size_t index = 0; index < dimension; ++index) {
      result.update[index] += alpha * workspace.direction[index];
      workspace.residual[index] -= alpha * workspace.product[index];
    }
    result.iterations = iteration + 1;
    const double residual_norm = std::sqrt(std::max(0.0,
      dot(workspace.residual, workspace.residual)));
    if(residual_norm <= options.pcg_tolerance * std::max(1.0, initial_norm)) {
      result.stop = termination::converged_gradient;
      result.converged = true;
      return result;
    }
    apply_preconditioner();
    const double next_projection = dot(workspace.residual, workspace.preconditioned);
    if(!std::isfinite(next_projection) || residual_projection == 0.0) {
      result.stop = termination::numerical_breakdown;
      return result;
    }
    const double beta = next_projection / residual_projection;
    for(std::size_t index = 0; index < dimension; ++index)
      workspace.direction[index] = workspace.preconditioned[index] +
        beta * workspace.direction[index];
    residual_projection = next_projection;
  }
  result.stop = termination::maximum_iterations;
  return result;
}

}
