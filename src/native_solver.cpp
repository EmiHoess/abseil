#include "internal.hpp"
#include "linear_solver.hpp"
#include "sparse_system.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace seil::detail {
namespace {

operation_error option_error(std::string message)
{
  return {operation_code::invalid_argument, operation_stage::validation, 0,
    std::move(message)};
}

bool valid_options(const common_options& common, const native_lm_options& lm)
{
  return common.maximum_iterations > 0 && common.maximum_seconds > 0.0 &&
    std::isfinite(common.maximum_seconds) && lm.maximum_rejected_steps > 0 &&
    lm.pcg_maximum_iterations > 0 && lm.initial_damping > 0.0 &&
    lm.minimum_damping > 0.0 && lm.maximum_damping >= lm.minimum_damping &&
    lm.initial_damping >= lm.minimum_damping &&
    lm.initial_damping <= lm.maximum_damping && lm.minimum_cost_change >= 0.0 &&
    lm.minimum_gradient_norm >= 0.0 && lm.minimum_update_norm >= 0.0 &&
    lm.maximum_translation_step_m > 0.0 && lm.maximum_yaw_step_rad > 0.0 &&
    lm.pcg_tolerance > 0.0 && std::isfinite(lm.initial_damping) &&
    std::isfinite(lm.minimum_damping) && std::isfinite(lm.maximum_damping) &&
    std::isfinite(lm.minimum_cost_change) &&
    std::isfinite(lm.minimum_gradient_norm) &&
    std::isfinite(lm.minimum_update_norm) &&
    std::isfinite(lm.maximum_translation_step_m) &&
    std::isfinite(lm.maximum_yaw_step_rad) && std::isfinite(lm.pcg_tolerance);
}

double vector_norm(const std::vector<double>& value)
{
  double squared = 0.0;
  for(double item : value)
    squared += item * item;
  return std::sqrt(std::max(0.0, squared));
}

double apply_update(
  const topology_index& topology,
  const native_lm_options& options,
  const std::vector<double>& update,
  std::vector<pose4>& poses
)
{
  double squared_norm = 0.0;
  for(std::size_t node_index = 0; node_index < poses.size(); ++node_index) {
    const std::int32_t active = topology.active_node[node_index];
    if(active < 0)
      continue;
    const std::size_t offset = static_cast<std::size_t>(active) * 4U;
    double dx = update[offset];
    double dy = update[offset + 1];
    double dz = update[offset + 2];
    double dyaw = std::clamp(update[offset + 3],
      -options.maximum_yaw_step_rad, options.maximum_yaw_step_rad);
    const double translation = std::sqrt(dx * dx + dy * dy + dz * dz);
    if(translation > options.maximum_translation_step_m) {
      const double scale = options.maximum_translation_step_m / translation;
      dx *= scale;
      dy *= scale;
      dz *= scale;
    }
    poses[node_index].x += dx;
    poses[node_index].y += dy;
    poses[node_index].z += dz;
    poses[node_index].yaw = wrap_yaw(poses[node_index].yaw + dyaw);
    squared_norm += dx * dx + dy * dy + dz * dz + dyaw * dyaw;
  }
  return std::sqrt(std::max(0.0, squared_norm));
}

std::string termination_message(termination value)
{
  switch(value) {
    case termination::converged_cost: return "cost tolerance reached";
    case termination::converged_gradient: return "gradient tolerance reached";
    case termination::converged_update: return "update tolerance reached";
    case termination::maximum_time: return "maximum solve time reached";
    case termination::rejection_limit: return "rejected step limit reached";
    default: return "native termination condition reached";
  }
}

}

std::optional<termination> native_termination(
  const native_termination_state& state,
  const common_options& common,
  const native_lm_options& options
) noexcept
{
  if(state.elapsed_seconds >= common.maximum_seconds)
    return termination::maximum_time;
  if(state.gradient_norm <= options.minimum_gradient_norm)
    return termination::converged_gradient;
  if(!state.step_complete)
    return std::nullopt;
  if(state.accepted && state.update_norm <= options.minimum_update_norm)
    return termination::converged_update;
  if(state.accepted && state.cost_change <= options.minimum_cost_change)
    return termination::converged_cost;
  if(!state.accepted && state.rejected_steps >= options.maximum_rejected_steps)
    return termination::rejection_limit;
  return std::nullopt;
}

operation_result<solve_result> solve_native_cpu(
  const prepared_problem& value,
  const common_options& common,
  const native_cpu_options& options
)
{
  if(!valid_options(common, options.lm))
    return std::unexpected(option_error("native solver options are invalid"));
  const auto* storage = prepared_problem_access::get(value);
  if(storage == nullptr)
    return std::unexpected(option_error("problem is not prepared"));

  solve_result result;
  result.selected_backend = backend::native_cpu;
  result.poses.reserve(storage->value.nodes.size());
  for(const auto& node : storage->value.nodes)
    result.poses.push_back(node.initial);

  auto current = build_sparse_system(*storage, result.poses);
  if(!current)
    return std::unexpected(std::move(current.error()));
  result.initial_cost = current->cost;
  result.final_cost = current->cost;
  result.registration = current->registration;
  if(storage->topology.active_count == 0) {
    result.stop = termination::no_free_nodes;
    result.solution_usable = true;
    result.message = "problem has no free nodes";
    return result;
  }
  if(storage->topology.unanchored_components != 0) {
    result.stop = termination::singular_graph;
    result.message = "pose graph has an unanchored component";
    return result;
  }

  const auto started = std::chrono::steady_clock::now();
  linear_workspace workspace;
  double damping = options.lm.initial_damping;
  std::size_t rejected_steps = 0;
  for(std::size_t iteration = 0; iteration < common.maximum_iterations; ++iteration) {
    const double elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
    const double gradient_norm = vector_norm(current->gradient);
    native_termination_state state{
      elapsed, gradient_norm, 0.0, 0.0, rejected_steps, false, false
    };
    if(const auto stop = native_termination(state, common, options.lm)) {
      result.stop = *stop;
      result.solution_usable = true;
      result.message = termination_message(*stop);
      return result;
    }

    linear_solution step = solve_sparse_system(
      storage->topology, *current, damping, options.lm, workspace);
    if(!step.converged) {
      result.stop = step.stop;
      result.message = "linear solve failed";
      return result;
    }
    std::vector<pose4> candidate = result.poses;
    const double update_norm = apply_update(
      storage->topology, options.lm, step.update, candidate);
    if(!std::isfinite(update_norm)) {
      result.stop = termination::numerical_breakdown;
      result.message = "pose update is not finite";
      return result;
    }
    auto candidate_equations = build_sparse_system(*storage, candidate);
    if(!candidate_equations)
      return std::unexpected(std::move(candidate_equations.error()));
    const double cost_before = current->cost.total();
    const double cost_after = candidate_equations->cost.total();
    if(cost_after < cost_before) {
      result.poses = std::move(candidate);
      current = std::move(candidate_equations);
      result.final_cost = current->cost;
      result.registration = current->registration;
      damping = std::max(options.lm.minimum_damping, damping * 0.3333333333333333);
      rejected_steps = 0;
      state.update_norm = update_norm;
      state.cost_change = cost_before - cost_after;
      state.step_complete = true;
      state.accepted = true;
      if(const auto stop = native_termination(state, common, options.lm)) {
        result.stop = *stop;
        result.solution_usable = true;
        result.message = termination_message(*stop);
        return result;
      }
    } else {
      damping = std::min(options.lm.maximum_damping, damping * 10.0);
      ++rejected_steps;
      state.update_norm = update_norm;
      state.cost_change = cost_before - cost_after;
      state.rejected_steps = rejected_steps;
      state.step_complete = true;
      if(const auto stop = native_termination(state, common, options.lm)) {
        result.stop = *stop;
        result.solution_usable = true;
        result.message = termination_message(*stop);
        return result;
      }
    }
  }
  result.stop = termination::maximum_iterations;
  result.solution_usable = true;
  result.message = "maximum iteration count reached";
  return result;
}

}
