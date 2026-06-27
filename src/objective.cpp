#include "internal.hpp"
#include "sparse_system.hpp"

#include <abseil/objective.hpp>

#include <algorithm>
#include <cmath>
#include <utility>

namespace abseil::detail {

double robust_cost(const loss& robust_loss, double residual) noexcept
{
  const double absolute = std::abs(residual);
  switch(robust_loss.kind) {
    case loss_kind::huber:
      return absolute <= robust_loss.scale ? 0.5 * residual * residual :
        robust_loss.scale * (absolute - 0.5 * robust_loss.scale);
    case loss_kind::cauchy: {
      const double normalized = residual / robust_loss.scale;
      return 0.5 * robust_loss.scale * robust_loss.scale *
        std::log1p(normalized * normalized);
    }
    case loss_kind::none:
      return 0.5 * residual * residual;
  }
  return 0.0;
}

double robust_weight(const loss& robust_loss, double residual) noexcept
{
  const double absolute = std::abs(residual);
  switch(robust_loss.kind) {
    case loss_kind::huber:
      return absolute <= robust_loss.scale || absolute == 0.0 ? 1.0 :
        robust_loss.scale / absolute;
    case loss_kind::cauchy: {
      const double normalized = residual / robust_loss.scale;
      return 1.0 / (1.0 + normalized * normalized);
    }
    case loss_kind::none:
      return 1.0;
  }
  return 0.0;
}

pose_factor_row evaluate_pose_factor(
  const pose4& left,
  const pose4& right,
  const pose4& measured,
  const matrix4& sqrt_information,
  const loss& robust_loss
) noexcept
{
  const double cosine = std::cos(left.yaw);
  const double sine = std::sin(left.yaw);
  const pose4 predicted = relative_pose(left, right);
  const double px = predicted.x;
  const double py = predicted.y;
  const double raw[4]{
    px - measured.x,
    py - measured.y,
    predicted.z - measured.z,
    wrap_yaw(predicted.yaw - measured.yaw)
  };
  const double raw_i[16]{
    -cosine, -sine, 0.0, py,
    sine, -cosine, 0.0, -px,
    0.0, 0.0, -1.0, 0.0,
    0.0, 0.0, 0.0, -1.0
  };
  const double raw_j[16]{
    cosine, sine, 0.0, 0.0,
    -sine, cosine, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0
  };

  pose_factor_row output;
  double squared_norm = 0.0;
  for(std::size_t row = 0; row < 4; ++row) {
    for(std::size_t inner = 0; inner < 4; ++inner) {
      output.residual[row] += sqrt_information[row * 4U + inner] * raw[inner];
      for(std::size_t column = 0; column < 4; ++column) {
        output.jacobian_i[row * 4U + column] +=
          sqrt_information[row * 4U + inner] * raw_i[inner * 4U + column];
        output.jacobian_j[row * 4U + column] +=
          sqrt_information[row * 4U + inner] * raw_j[inner * 4U + column];
      }
    }
    squared_norm += output.residual[row] * output.residual[row];
  }
  const double norm = std::sqrt(std::max(0.0, squared_norm));
  output.cost = robust_cost(robust_loss, norm);
  output.weight = robust_weight(robust_loss, norm);
  return output;
}

std::vector<pose_factor_row> evaluate_relative_factors(
  const prepared_storage& storage,
  std::span<const pose4> poses
)
{
  std::vector<pose_factor_row> rows;
  rows.reserve(storage.value.relative_factors.size());
  for(const auto& factor : storage.value.relative_factors) {
    pose_factor_row row = evaluate_pose_factor(
      poses[factor.node_i], poses[factor.node_j], factor.measured,
      factor.sqrt_information, factor.robust_loss);
    row.active_i = storage.topology.active_node[factor.node_i];
    row.active_j = storage.topology.active_node[factor.node_j];
    rows.push_back(std::move(row));
  }
  return rows;
}

std::vector<pose_factor_row> evaluate_absolute_factors(
  const prepared_storage& storage,
  std::span<const pose4> poses
)
{
  std::vector<pose_factor_row> rows;
  rows.reserve(storage.value.absolute_factors.size());
  for(const auto& factor : storage.value.absolute_factors) {
    pose_factor_row row = evaluate_pose_factor(
      poses[factor.reference_node], poses[factor.measured_node],
      factor.measured_relative, factor.sqrt_information, factor.robust_loss);
    row.active_i = storage.topology.active_node[factor.reference_node];
    row.active_j = storage.topology.active_node[factor.measured_node];
    rows.push_back(std::move(row));
  }
  return rows;
}

std::vector<registration_row> evaluate_registration_group(
  const prepared_storage& storage,
  const registration_group& group,
  std::span<const pose4> poses,
  double& total_confidence,
  registration_statistics* statistics
)
{
  std::vector<registration_row> rows(group.sample_count);
  total_confidence = 0.0;
  const auto& model = storage.value.registration;
  const bool uniform = model.profile == registration_profile::uniform_trilinear;
  const pose4& reference = poses[group.node_i];
  const pose4& reading = poses[group.node_j];

  for(std::size_t local = 0; local < group.sample_count; ++local) {
    const auto& sample = storage.value.registration_samples[group.sample_begin + local];
    auto& row = rows[local];
    row.active_i = storage.topology.active_node[group.node_i];
    row.active_j = storage.topology.active_node[group.node_j];
    if(!std::isfinite(sample.reference_distance) ||
      !std::isfinite(sample.reference_tsdf_weight) ||
      (!uniform && sample.reference_tsdf_weight < model.minimum_tsdf_weight))
      row.rejection = 2;

    const double reading_cosine = std::cos(reading.yaw);
    const double reading_sine = std::sin(reading.yaw);
    const vector3 point = transform_point_between(
      reference, reading, sample.reference_local_point);

    field_query confidence;
    if(!uniform && row.rejection == 0 &&
      !sample_field(storage, field_kind::tsdf, group.volume_j, point,
        field_interpolation::trilinear, true, confidence)) {
      if(!model.allow_nearest_fallback ||
        !sample_field(storage, field_kind::tsdf, group.volume_j, point,
          field_interpolation::nearest, false, confidence))
        row.rejection = 4;
    }
    if(!uniform && row.rejection == 0 &&
      confidence.sample.weight < model.minimum_tsdf_weight)
      row.rejection = 3;

    field_query query;
    if(row.rejection == 0 &&
      !sample_field(storage, group.field, group.volume_j, point,
        field_interpolation::trilinear, true, query)) {
      if(uniform || !model.allow_nearest_fallback ||
        !sample_field(storage, group.field, group.volume_j, point,
          field_interpolation::nearest, true, query))
        row.rejection = 5;
    }
    if(row.rejection == 0 && !query.gradient_valid)
      row.rejection = 6;
    const double gx = query.gradient.x;
    const double gy = query.gradient.y;
    const double gz = query.gradient.z;
    const double gradient_norm = std::sqrt(gx * gx + gy * gy + gz * gz);
    if(!uniform && row.rejection == 0 &&
      !(gradient_norm >= model.minimum_gradient_norm))
      row.rejection = 6;

    row.residual = sample.reference_distance - query.sample.distance;
    if(row.rejection == 0 && model.maximum_absolute_residual_m > 0.0 &&
      std::abs(row.residual) > model.maximum_absolute_residual_m)
      row.rejection = 7;
    if(row.rejection == 0) {
      const vector3 yaw_world = rotate_yaw_derivative(
        sample.reference_local_point, reference.yaw);
      const vector3 yaw_reading = rotate_yaw(yaw_world, -reading.yaw);
      row.jacobian_i = {
        -(gx * reading_cosine - gy * reading_sine),
        -(gx * reading_sine + gy * reading_cosine),
        -gz,
        -(gx * yaw_reading.x + gy * yaw_reading.y)
      };
      row.jacobian_j = {
        -row.jacobian_i[0],
        -row.jacobian_i[1],
        -row.jacobian_i[2],
        -(gx * point.y - gy * point.x)
      };
      row.confidence = uniform ? 1.0 :
        std::min(model.maximum_confidence_weight,
          std::min(sample.reference_tsdf_weight, confidence.sample.weight));
      bool finite = std::isfinite(row.residual) && std::isfinite(row.confidence) &&
        row.confidence > 0.0;
      for(double item : row.jacobian_i)
        finite = finite && std::isfinite(item);
      for(double item : row.jacobian_j)
        finite = finite && std::isfinite(item);
      if(!finite)
        row.rejection = 8;
    }

    if(row.rejection == 0) {
      total_confidence += row.confidence;
      if(statistics != nullptr)
        ++statistics->accepted_rows;
    } else if(statistics != nullptr && row.rejection < statistics->rejections.size()) {
      ++statistics->rejections[row.rejection];
    }
  }
  return rows;
}

namespace {

void add_matrix_gradient(
  std::vector<double>& gradient,
  std::int32_t active,
  const std::array<double, 16>& jacobian,
  const std::array<double, 4>& residual,
  double weight
)
{
  if(active < 0)
    return;
  for(std::size_t column = 0; column < 4; ++column)
    for(std::size_t row = 0; row < 4; ++row)
      gradient[static_cast<std::size_t>(active) * 4U + column] +=
        weight * jacobian[row * 4U + column] * residual[row];
}

void add_matrix_outer(
  std::vector<double>& hessian,
  std::size_t dimension,
  std::int32_t active_i,
  std::int32_t active_j,
  const std::array<double, 16>& left,
  const std::array<double, 16>& right,
  double weight
)
{
  if(active_i < 0 || active_j < 0)
    return;
  const std::size_t row_offset = static_cast<std::size_t>(active_i) * 4U;
  const std::size_t column_offset = static_cast<std::size_t>(active_j) * 4U;
  for(std::size_t row = 0; row < 4; ++row)
    for(std::size_t column = 0; column < 4; ++column)
      for(std::size_t inner = 0; inner < 4; ++inner)
        hessian[(row_offset + row) * dimension + column_offset + column] +=
          weight * left[inner * 4U + row] * right[inner * 4U + column];
}

void add_pose_row(
  objective_equations& equations,
  const pose_factor_row& row,
  bool derivatives,
  std::size_t dimension
)
{
  if(!derivatives || !(row.weight > 0.0))
    return;
  add_matrix_outer(equations.dense_hessian, dimension, row.active_i, row.active_i,
    row.jacobian_i, row.jacobian_i, row.weight);
  add_matrix_outer(equations.dense_hessian, dimension, row.active_j, row.active_j,
    row.jacobian_j, row.jacobian_j, row.weight);
  add_matrix_outer(equations.dense_hessian, dimension, row.active_i, row.active_j,
    row.jacobian_i, row.jacobian_j, row.weight);
  add_matrix_outer(equations.dense_hessian, dimension, row.active_j, row.active_i,
    row.jacobian_j, row.jacobian_i, row.weight);
  add_matrix_gradient(equations.gradient, row.active_i, row.jacobian_i,
    row.residual, row.weight);
  add_matrix_gradient(equations.gradient, row.active_j, row.jacobian_j,
    row.residual, row.weight);
}

void add_scalar_row(
  objective_equations& equations,
  const registration_row& row,
  double weight,
  std::size_t dimension
)
{
  const auto add_gradient = [&](std::int32_t active,
    const std::array<double, 4>& jacobian) {
    if(active < 0)
      return;
    for(std::size_t axis = 0; axis < 4; ++axis)
      equations.gradient[static_cast<std::size_t>(active) * 4U + axis] +=
        weight * jacobian[axis] * row.residual;
  };
  const auto add_outer = [&](std::int32_t active_i, std::int32_t active_j,
    const std::array<double, 4>& left, const std::array<double, 4>& right) {
    if(active_i < 0 || active_j < 0)
      return;
    const std::size_t row_offset = static_cast<std::size_t>(active_i) * 4U;
    const std::size_t column_offset = static_cast<std::size_t>(active_j) * 4U;
    for(std::size_t i = 0; i < 4; ++i)
      for(std::size_t j = 0; j < 4; ++j)
        equations.dense_hessian[(row_offset + i) * dimension + column_offset + j] +=
          weight * left[i] * right[j];
  };
  add_gradient(row.active_i, row.jacobian_i);
  add_gradient(row.active_j, row.jacobian_j);
  add_outer(row.active_i, row.active_i, row.jacobian_i, row.jacobian_i);
  add_outer(row.active_j, row.active_j, row.jacobian_j, row.jacobian_j);
  add_outer(row.active_i, row.active_j, row.jacobian_i, row.jacobian_j);
  add_outer(row.active_j, row.active_i, row.jacobian_j, row.jacobian_i);
}

}

operation_result<objective_equations> assemble_objective(
  const prepared_storage& storage,
  std::span<const pose4> poses,
  bool with_derivatives
)
{
  if(poses.size() != storage.value.nodes.size())
    return std::unexpected(operation_error{operation_code::invalid_argument,
      operation_stage::validation, 0, "pose count does not match prepared problem"});
  if(!std::all_of(poses.begin(), poses.end(), finite_pose))
    return std::unexpected(operation_error{operation_code::invalid_argument,
      operation_stage::validation, 0, "pose array contains non-finite values"});

  objective_equations equations;
  const std::size_t dimension = storage.topology.active_count * 4U;
  if(with_derivatives) {
    equations.gradient.assign(dimension, 0.0);
    equations.dense_hessian.assign(dimension * dimension, 0.0);
  }

  for(const auto& row : evaluate_relative_factors(storage, poses)) {
    equations.cost.relative += row.cost;
    add_pose_row(equations, row, with_derivatives, dimension);
  }
  for(const auto& row : evaluate_absolute_factors(storage, poses)) {
    equations.cost.absolute += row.cost;
    add_pose_row(equations, row, with_derivatives, dimension);
  }
  for(const auto& group : storage.value.registration_groups) {
    double total_confidence = 0.0;
    const auto rows = evaluate_registration_group(
      storage, group, poses, total_confidence, &equations.registration);
    const bool uniform = storage.value.registration.profile ==
      registration_profile::uniform_trilinear;
    for(const auto& row : rows) {
      if(row.rejection != 0)
        continue;
      const double confidence = uniform ? row.confidence :
        row.confidence / total_confidence;
      const double loss_weight = robust_weight(
        storage.value.registration.robust_loss, row.residual);
      const double weight = group.direction_weight * confidence * loss_weight;
      equations.cost.registration += group.direction_weight * confidence *
        robust_cost(storage.value.registration.robust_loss, row.residual);
      if(with_derivatives)
        add_scalar_row(equations, row, weight, dimension);
    }
  }
  return equations;
}

}

namespace abseil {

operation_result<robust_evaluation> evaluate_loss(
  const loss& robust_loss,
  double residual
)
{
  if(!std::isfinite(residual))
    return std::unexpected(operation_error{
      operation_code::invalid_argument,
      operation_stage::validation,
      0,
      "robust residual must be finite"
    });
  if(robust_loss.kind != loss_kind::none &&
    (!(robust_loss.scale > 0.0) || !std::isfinite(robust_loss.scale)))
    return std::unexpected(operation_error{
      operation_code::invalid_argument,
      operation_stage::validation,
      0,
      "robust loss scale must be finite and positive"
    });
  return robust_evaluation{
    detail::robust_cost(robust_loss, residual),
    detail::robust_weight(robust_loss, residual)
  };
}

operation_result<objective_snapshot> evaluate(
  const prepared_problem& value,
  std::span<const pose4> poses
)
{
  const auto* storage = detail::prepared_problem_access::get(value);
  if(storage == nullptr)
    return std::unexpected(operation_error{operation_code::invalid_argument,
      operation_stage::validation, 0, "problem is not prepared"});
  auto equations = detail::assemble_objective(*storage, poses, false);
  if(!equations)
    return std::unexpected(std::move(equations.error()));
  return objective_snapshot{equations->cost, equations->registration};
}

operation_result<linearization_snapshot> linearize(
  const prepared_problem& value,
  std::span<const pose4> poses
)
{
  const auto* storage = detail::prepared_problem_access::get(value);
  if(storage == nullptr)
    return std::unexpected(operation_error{operation_code::invalid_argument,
      operation_stage::validation, 0, "problem is not prepared"});
  auto system = detail::build_sparse_system(*storage, poses);
  if(!system)
    return std::unexpected(std::move(system.error()));

  linearization_snapshot output;
  output.cost = system->cost;
  output.registration = system->registration;
  output.hessian_blocks = std::move(system->hessian);
  output.gradient = std::move(system->gradient);
  output.sparse_blocks.reserve(system->blocks.size());
  for(const auto& block : system->blocks)
    output.sparse_blocks.push_back({block.row, block.column});
  return output;
}

}
