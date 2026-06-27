#include "internal.hpp"

#include <abseil/objective.hpp>

#include <ceres/ceres.h>
#include <ceres/version.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace abseil::detail {
namespace {

class pose4_manifold final : public ceres::Manifold {
public:
  int AmbientSize() const override
  {
    return 4;
  }

  int TangentSize() const override
  {
    return 4;
  }

  bool Plus(const double* value, const double* delta, double* output) const override
  {
    output[0] = value[0] + delta[0];
    output[1] = value[1] + delta[1];
    output[2] = value[2] + delta[2];
    output[3] = wrap_yaw(value[3] + delta[3]);
    return true;
  }

  bool PlusJacobian(const double*, double* jacobian) const override
  {
    std::fill(jacobian, jacobian + 16, 0.0);
    jacobian[0] = jacobian[5] = jacobian[10] = jacobian[15] = 1.0;
    return true;
  }

  bool Minus(const double* left, const double* right, double* output) const override
  {
    output[0] = left[0] - right[0];
    output[1] = left[1] - right[1];
    output[2] = left[2] - right[2];
    output[3] = wrap_yaw(left[3] - right[3]);
    return true;
  }

  bool MinusJacobian(const double*, double* jacobian) const override
  {
    std::fill(jacobian, jacobian + 16, 0.0);
    jacobian[0] = jacobian[5] = jacobian[10] = jacobian[15] = 1.0;
    return true;
  }
};

struct relative_pose_cost {
  pose4 measured{};
  matrix4 sqrt_information{};

  template<class scalar_type>
  bool operator()(
    const scalar_type* const left,
    const scalar_type* const right,
    scalar_type* residual
  ) const
  {
    using ceres::atan2;
    using ceres::cos;
    using ceres::sin;
    const scalar_type dx = right[0] - left[0];
    const scalar_type dy = right[1] - left[1];
    const scalar_type cosine = cos(left[3]);
    const scalar_type sine = sin(left[3]);
    const scalar_type raw[4]{
      cosine * dx + sine * dy - scalar_type(measured.x),
      -sine * dx + cosine * dy - scalar_type(measured.y),
      right[2] - left[2] - scalar_type(measured.z),
      atan2(sin(right[3] - left[3] - scalar_type(measured.yaw)),
        cos(right[3] - left[3] - scalar_type(measured.yaw)))
    };
    for(std::size_t row = 0; row < 4; ++row) {
      residual[row] = scalar_type(0.0);
      for(std::size_t column = 0; column < 4; ++column)
        residual[row] += scalar_type(sqrt_information[row * 4U + column]) *
          raw[column];
    }
    return true;
  }
};

ceres::LossFunction* make_loss(const loss& robust_loss)
{
  switch(robust_loss.kind) {
    case loss_kind::none: return nullptr;
    case loss_kind::huber: return new ceres::HuberLoss(robust_loss.scale);
    case loss_kind::cauchy: return new ceres::CauchyLoss(robust_loss.scale);
  }
  return nullptr;
}

class registration_cost final : public ceres::CostFunction {
public:
  registration_cost(const prepared_storage& storage, std::size_t group_index)
    : storage_(&storage), group_index_(group_index)
  {
    const auto& group = storage.value.registration_groups[group_index];
    set_num_residuals(static_cast<int>(group.sample_count));
    mutable_parameter_block_sizes()->push_back(4);
    mutable_parameter_block_sizes()->push_back(4);
    base_poses_.reserve(storage.value.nodes.size());
    for(const auto& node : storage.value.nodes)
      base_poses_.push_back(node.initial);
  }

  bool Evaluate(
    double const* const* parameters,
    double* residuals,
    double** jacobians
  ) const override
  {
    const auto& group = storage_->value.registration_groups[group_index_];
    std::vector<pose4> poses = base_poses_;
    poses[group.node_i] = {
      parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3]
    };
    poses[group.node_j] = {
      parameters[1][0], parameters[1][1], parameters[1][2], parameters[1][3]
    };
    double total_confidence = 0.0;
    const auto rows = evaluate_registration_group(
      *storage_, group, poses, total_confidence, nullptr);
    const bool uniform = storage_->value.registration.profile ==
      registration_profile::uniform_trilinear;
    for(std::size_t index = 0; index < rows.size(); ++index) {
      const auto& row = rows[index];
      residuals[index] = 0.0;
      double derivative = 0.0;
      if(row.rejection == 0 && total_confidence > 0.0) {
        const double confidence = uniform ? row.confidence :
          row.confidence / total_confidence;
        const double factor = group.direction_weight * confidence;
        const double cost = robust_cost(
          storage_->value.registration.robust_loss, row.residual);
        const double weighted_cost = factor * cost;
        if(weighted_cost > 0.0 && std::isfinite(weighted_cost)) {
          residuals[index] = std::copysign(
            std::sqrt(2.0 * weighted_cost), row.residual);
          derivative = factor * robust_weight(
            storage_->value.registration.robust_loss, row.residual) *
            row.residual / residuals[index];
        } else if(cost == 0.0 && factor >= 0.0) {
          derivative = std::sqrt(factor);
        }
      }
      if(jacobians != nullptr && jacobians[0] != nullptr)
        for(std::size_t axis = 0; axis < 4; ++axis)
          jacobians[0][index * 4U + axis] = derivative * row.jacobian_i[axis];
      if(jacobians != nullptr && jacobians[1] != nullptr)
        for(std::size_t axis = 0; axis < 4; ++axis)
          jacobians[1][index * 4U + axis] = derivative * row.jacobian_j[axis];
    }
    return true;
  }

private:
  const prepared_storage* storage_{};
  std::size_t group_index_{};
  std::vector<pose4> base_poses_;
};

class history_callback final : public ceres::IterationCallback {
public:
  explicit history_callback(std::vector<iteration_summary>& history)
    : history_(&history)
  {
  }

  ceres::CallbackReturnType operator()(
    const ceres::IterationSummary& value
  ) override
  {
    if(value.iteration == 0)
      return ceres::SOLVER_CONTINUE;
    history_->push_back({
      history_->size(),
      static_cast<std::size_t>(std::max(0, value.linear_solver_iterations)),
      value.step_is_successful,
      value.cost + value.cost_change,
      value.cost,
      0.0,
      value.trust_region_radius,
      value.gradient_norm,
      value.step_norm,
      value.iteration_time_in_seconds,
      value.step_solver_time_in_seconds
    });
    return ceres::SOLVER_CONTINUE;
  }

private:
  std::vector<iteration_summary>* history_{};
};

termination map_termination(const ceres::Solver::Summary& summary)
{
  if(summary.termination_type == ceres::FAILURE ||
    summary.termination_type == ceres::USER_FAILURE)
    return termination::numerical_breakdown;
  if(summary.termination_type == ceres::NO_CONVERGENCE) {
    if(summary.message.find("Maximum solver time") != std::string::npos)
      return termination::maximum_time;
    return termination::maximum_iterations;
  }
  if(summary.message.find("Gradient tolerance") != std::string::npos)
    return termination::converged_gradient;
  if(summary.message.find("Parameter tolerance") != std::string::npos)
    return termination::converged_update;
  return termination::converged_cost;
}

bool valid_options(const common_options& common, const ceres_cpu_options& options)
{
  return common.maximum_iterations > 0 && common.maximum_seconds > 0.0 &&
    options.threads > 0 && options.function_tolerance >= 0.0 &&
    options.gradient_tolerance >= 0.0 && options.parameter_tolerance >= 0.0 &&
    options.minimum_damping > 0.0 &&
    options.initial_damping >= options.minimum_damping &&
    options.maximum_damping >= options.initial_damping;
}

}

bool ceres_backend_available() noexcept
{
  return true;
}

operation_result<solve_result> solve_ceres_cpu(
  const prepared_problem& prepared,
  const common_options& common,
  const ceres_cpu_options& options
)
{
  if(!valid_options(common, options))
    return std::unexpected(operation_error{operation_code::invalid_argument,
      operation_stage::validation, 0, "ceres solver options are invalid"});
  const auto* storage = prepared_problem_access::get(prepared);
  if(storage == nullptr)
    return std::unexpected(operation_error{operation_code::invalid_argument,
      operation_stage::validation, 0, "problem is not prepared"});
  if(storage->fields)
    return std::unexpected(operation_error{operation_code::invalid_argument,
      operation_stage::validation, 0,
      "ceres_cpu does not accept borrowed field snapshots"});

  const auto started = std::chrono::steady_clock::now();
  solve_result result;
  result.selected_backend = backend::ceres_cpu;
  result.telemetry.implementation = std::string("ceres_") + CERES_VERSION_STRING;
  result.telemetry.device = "host";
  result.poses.reserve(storage->value.nodes.size());
  for(const auto& node : storage->value.nodes)
    result.poses.push_back(node.initial);
  auto initial = evaluate(prepared, result.poses);
  if(!initial)
    return std::unexpected(std::move(initial.error()));
  result.initial_cost = initial->cost;
  result.final_cost = initial->cost;
  result.registration = initial->registration;
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

  std::vector<std::array<double, 4>> parameters;
  parameters.reserve(result.poses.size());
  for(const auto& pose : result.poses)
    parameters.push_back({pose.x, pose.y, pose.z, pose.yaw});
  ceres::Problem problem;
  for(std::size_t index = 0; index < parameters.size(); ++index) {
    problem.AddParameterBlock(parameters[index].data(), 4);
    problem.SetManifold(parameters[index].data(), new pose4_manifold());
    if(storage->value.nodes[index].constant)
      problem.SetParameterBlockConstant(parameters[index].data());
  }
  const auto add_factor = [&](std::size_t left, std::size_t right,
    const pose4& measured, const matrix4& sqrt_information,
    const loss& robust_loss) {
    auto* cost = new ceres::AutoDiffCostFunction<relative_pose_cost, 4, 4, 4>(
      new relative_pose_cost{measured, sqrt_information});
    problem.AddResidualBlock(cost, make_loss(robust_loss),
      parameters[left].data(), parameters[right].data());
  };
  for(const auto& factor : storage->value.relative_factors)
    add_factor(factor.node_i, factor.node_j, factor.measured,
      factor.sqrt_information, factor.robust_loss);
  for(const auto& factor : storage->value.absolute_factors)
    add_factor(factor.reference_node, factor.measured_node,
      factor.measured_relative, factor.sqrt_information, factor.robust_loss);
  for(std::size_t index = 0;
      index < storage->value.registration_groups.size(); ++index) {
    const auto& group = storage->value.registration_groups[index];
    if(group.sample_count == 0)
      continue;
    problem.AddResidualBlock(new registration_cost(*storage, index), nullptr,
      parameters[group.node_i].data(), parameters[group.node_j].data());
  }

  history_callback callback(result.iterations);
  ceres::Solver::Options solver_options;
  solver_options.minimizer_type = ceres::TRUST_REGION;
  solver_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  solver_options.max_num_iterations = static_cast<int>(std::min<std::size_t>(
    common.maximum_iterations, static_cast<std::size_t>(std::numeric_limits<int>::max())));
  solver_options.max_solver_time_in_seconds = common.maximum_seconds;
  solver_options.function_tolerance = options.function_tolerance;
  solver_options.gradient_tolerance = options.gradient_tolerance;
  solver_options.parameter_tolerance = options.parameter_tolerance;
  solver_options.initial_trust_region_radius = 1.0 / options.initial_damping;
  solver_options.min_trust_region_radius = 1.0 / options.maximum_damping;
  solver_options.max_trust_region_radius = 1.0 / options.minimum_damping;
  solver_options.num_threads = static_cast<int>(std::min<std::size_t>(
    options.threads, static_cast<std::size_t>(std::numeric_limits<int>::max())));
  solver_options.update_state_every_iteration = true;
  solver_options.callbacks.push_back(&callback);
  solver_options.minimizer_progress_to_stdout = options.progress_to_stdout;
  solver_options.logging_type = options.progress_to_stdout ?
    ceres::PER_MINIMIZER_ITERATION : ceres::SILENT;

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);
  const bool finite = std::all_of(parameters.begin(), parameters.end(),
    [](const auto& pose) {
      return std::all_of(pose.begin(), pose.end(),
        [](double item) { return std::isfinite(item); });
    });
  result.solution_usable = summary.IsSolutionUsable() && finite;
  if(result.solution_usable)
    for(std::size_t index = 0; index < parameters.size(); ++index)
      result.poses[index] = {
        parameters[index][0], parameters[index][1], parameters[index][2],
        wrap_yaw(parameters[index][3])
      };
  auto final = evaluate(prepared, result.poses);
  if(!final)
    return std::unexpected(std::move(final.error()));
  result.final_cost = final->cost;
  result.registration = final->registration;
  result.stop = result.solution_usable ? map_termination(summary) :
    termination::numerical_breakdown;
  result.telemetry.evaluation_ms = 1000.0 *
    (std::max(0.0, summary.residual_evaluation_time_in_seconds) +
      std::max(0.0, summary.jacobian_evaluation_time_in_seconds));
  result.telemetry.linear_solver_ms = 1000.0 *
    std::max(0.0, summary.linear_solver_time_in_seconds);
  result.telemetry.total_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - started).count();
  for(const auto& iteration : result.iterations)
    if(iteration.accepted)
      ++result.telemetry.accepted_steps;
    else
      ++result.telemetry.rejected_steps;
  result.message = summary.BriefReport();
  return result;
}

}
