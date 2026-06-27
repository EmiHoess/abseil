#include <abseil/objective.hpp>

#include "../src/internal.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace {

void require(bool condition, const char* message)
{
  if(!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

bool near(double left, double right)
{
  return std::abs(left - right) < 1.0e-12;
}

std::array<double, 4> residual(
  const abseil::pose4& left,
  const abseil::pose4& right,
  const abseil::pose4& measured
)
{
  const auto row = abseil::detail::evaluate_pose_factor(
    left, right, measured, abseil::identity_matrix4(), {});
  return row.residual;
}

void adjust(abseil::pose4& value, std::size_t axis, double amount)
{
  switch(axis) {
    case 0: value.x += amount; break;
    case 1: value.y += amount; break;
    case 2: value.z += amount; break;
    case 3: value.yaw += amount; break;
    default: break;
  }
}

}

int main()
{
  auto plain = abseil::evaluate_loss({}, 2.0);
  require(plain.has_value(), "plain loss accepted");
  require(near(plain->cost, 2.0) && near(plain->weight, 1.0),
    "plain loss values");

  auto huber_edge = abseil::evaluate_loss({abseil::loss_kind::huber, 2.0}, 2.0);
  auto huber_tail = abseil::evaluate_loss({abseil::loss_kind::huber, 2.0}, -4.0);
  require(huber_edge.has_value() && near(huber_edge->cost, 2.0),
    "huber threshold cost");
  require(huber_tail.has_value() && near(huber_tail->cost, 6.0) &&
    near(huber_tail->weight, 0.5), "huber tail values");

  auto cauchy_zero = abseil::evaluate_loss({abseil::loss_kind::cauchy, 3.0}, 0.0);
  require(cauchy_zero.has_value() && near(cauchy_zero->cost, 0.0) &&
    near(cauchy_zero->weight, 1.0), "cauchy zero values");
  auto cauchy = abseil::evaluate_loss({abseil::loss_kind::cauchy, 2.0}, 2.0);
  require(cauchy.has_value() && near(cauchy->cost, 2.0 * std::log(2.0)) &&
    near(cauchy->weight, 0.5), "cauchy values");

  require(!abseil::evaluate_loss({abseil::loss_kind::huber, 0.0}, 1.0),
    "zero scale rejected");
  require(!abseil::evaluate_loss({}, std::numeric_limits<double>::infinity()),
    "infinite residual rejected");

  const abseil::pose4 left{1.0, -2.0, 0.5, 0.3};
  const abseil::pose4 right{4.0, 1.0, -0.5, -0.2};
  const abseil::pose4 measured{3.5, 1.8, -0.8, -0.4};
  const auto row = abseil::detail::evaluate_pose_factor(
    left, right, measured, abseil::identity_matrix4(),
    {abseil::loss_kind::huber, 2.0});
  require(row.cost > 0.0 && row.weight > 0.0 && row.weight <= 1.0,
    "relative row robust accounting");

  constexpr double step = 1.0e-6;
  for(std::size_t column = 0; column < 4; ++column) {
    abseil::pose4 moved = left;
    adjust(moved, column, step);
    const auto plus = residual(moved, right, measured);
    moved = left;
    adjust(moved, column, -step);
    const auto minus = residual(moved, right, measured);
    for(std::size_t output = 0; output < 4; ++output) {
      const double numerical = (plus[output] - minus[output]) / (2.0 * step);
      require(std::abs(numerical - row.jacobian_i[output * 4U + column]) < 1.0e-6,
        "left relative jacobian");
    }
  }
  for(std::size_t column = 0; column < 4; ++column) {
    abseil::pose4 moved = right;
    adjust(moved, column, step);
    const auto plus = residual(left, moved, measured);
    moved = right;
    adjust(moved, column, -step);
    const auto minus = residual(left, moved, measured);
    for(std::size_t output = 0; output < 4; ++output) {
      const double numerical = (plus[output] - minus[output]) / (2.0 * step);
      require(std::abs(numerical - row.jacobian_j[output * 4U + column]) < 1.0e-6,
        "right relative jacobian");
    }
  }

  abseil::problem graph;
  graph.nodes.push_back({1, left, true});
  graph.nodes.push_back({2, right, false});
  graph.relative_factors.push_back({0, 1, measured});
  auto prepared = abseil::prepare(graph);
  require(prepared.has_value(), "relative graph preparation");
  const auto* storage = abseil::detail::prepared_problem_access::get(*prepared);
  const std::array poses{left, right};
  const auto rows = abseil::detail::evaluate_relative_factors(*storage, poses);
  require(rows.size() == 1 && rows[0].active_i == -1 && rows[0].active_j == 0,
    "fixed gauge activity");

  graph.relative_factors.clear();
  graph.absolute_factors.push_back({0, 1, measured,
    abseil::identity_matrix4(2.0), {abseil::loss_kind::cauchy, 3.0}, 91});
  prepared = abseil::prepare(graph);
  require(prepared.has_value(), "absolute graph preparation");
  storage = abseil::detail::prepared_problem_access::get(*prepared);
  const auto absolute_rows = abseil::detail::evaluate_absolute_factors(*storage, poses);
  require(absolute_rows.size() == 1 && absolute_rows[0].active_i == -1 &&
    absolute_rows[0].active_j == 0, "absolute activity");
  const auto expected = abseil::detail::evaluate_pose_factor(left, right, measured,
    abseil::identity_matrix4(2.0), {abseil::loss_kind::cauchy, 3.0});
  require(near(absolute_rows[0].cost, expected.cost) &&
    near(absolute_rows[0].weight, expected.weight),
    "absolute loss accounting");
  for(std::size_t index = 0; index < expected.residual.size(); ++index)
    require(near(absolute_rows[0].residual[index], expected.residual[index]),
      "absolute residual direction");

  graph.absolute_factors.clear();
  graph.volumes.push_back({0, 1.0, true, true});
  graph.volumes.push_back({1, 1.0, true, true});
  abseil::tsdf_tile tsdf;
  abseil::esdf_tile esdf;
  tsdf.volume = 1;
  esdf.volume = 1;
  for(std::size_t z = 0; z < abseil::tile_edge; ++z)
    for(std::size_t y = 0; y < abseil::tile_edge; ++y)
      for(std::size_t x = 0; x < abseil::tile_edge; ++x) {
        const std::size_t index = x + 8U * (y + 8U * z);
        const float distance = static_cast<float>(x);
        tsdf.voxels[index] = {distance, 4.0F, 1U};
        esdf.voxels[index] = {distance, 1U};
      }
  graph.tsdf_tiles.push_back(tsdf);
  graph.esdf_tiles.push_back(esdf);
  graph.registration_samples.push_back({0, {2.5, 2.5, 2.5}, 2.0, 3.0, 1});
  graph.registration_groups.push_back({0, 1, 0, 1, abseil::field_kind::esdf,
    0, 1, 1.0, 7});
  graph.nodes[0].initial = {};
  graph.nodes[1].initial = {};
  graph.registration.profile = abseil::registration_profile::uniform_trilinear;
  prepared = abseil::prepare(graph);
  require(prepared.has_value(), "registration graph preparation");
  storage = abseil::detail::prepared_problem_access::get(*prepared);
  const std::array registration_poses{graph.nodes[0].initial, graph.nodes[1].initial};
  abseil::registration_statistics statistics;
  double confidence = 0.0;
  const auto registration_rows = abseil::detail::evaluate_registration_group(
    *storage, storage->value.registration_groups[0], registration_poses,
    confidence, &statistics);
  require(registration_rows.size() == 1 && registration_rows[0].rejection == 0,
    "registration row accepted");
  require(near(registration_rows[0].residual, 0.0) && near(confidence, 1.0) &&
    statistics.accepted_rows == 1, "registration row values");

  graph.registration.profile = abseil::registration_profile::confidence_weighted;
  graph.registration.maximum_confidence_weight = 2.5;
  prepared = abseil::prepare(graph);
  require(prepared.has_value(), "confidence graph preparation");
  storage = abseil::detail::prepared_problem_access::get(*prepared);
  statistics = {};
  const auto confidence_rows = abseil::detail::evaluate_registration_group(
    *storage, storage->value.registration_groups[0], registration_poses,
    confidence, &statistics);
  require(confidence_rows[0].rejection == 0 && near(confidence, 2.5),
    "confidence registration row");

  const abseil::pose4 source_pose{1.0, 2.0, 3.0, 0.2};
  const abseil::pose4 target_pose{-1.0, 0.5, 1.0, -0.4};
  const abseil::vector3 local_point{2.0, -3.0, 4.0};
  const auto world_point = abseil::transform_point(source_pose, local_point);
  const auto direct_point = abseil::transform_point_between(
    source_pose, target_pose, local_point);
  const auto composed_point = abseil::inverse_transform_point(target_pose, world_point);
  require(near(direct_point.x, composed_point.x) &&
    near(direct_point.y, composed_point.y) && near(direct_point.z, composed_point.z),
    "shared frame transformation");
  const auto derivative = abseil::rotate_yaw_derivative(local_point, source_pose.yaw);
  const auto plus_rotation = abseil::rotate_yaw(local_point, source_pose.yaw + step);
  const auto minus_rotation = abseil::rotate_yaw(local_point, source_pose.yaw - step);
  require(std::abs(derivative.x - (plus_rotation.x - minus_rotation.x) /
    (2.0 * step)) < 1.0e-6, "shared yaw derivative");

  auto assembled = abseil::detail::assemble_objective(
    *storage, registration_poses, true);
  require(assembled.has_value(), "objective assembly");
  require(assembled->cost.registration >= 0.0 &&
    assembled->registration.accepted_rows == 1, "objective split diagnostics");
  require(assembled->gradient.size() == 4 &&
    assembled->dense_hessian.size() == 16, "objective derivative dimensions");
  require(!abseil::detail::assemble_objective(*storage,
    std::span<const abseil::pose4>{}, false), "objective pose count validation");

  auto public_cost = abseil::evaluate(*prepared, registration_poses);
  require(public_cost.has_value() && public_cost->registration.accepted_rows == 1,
    "public objective diagnostics");
  auto public_system = abseil::linearize(*prepared, registration_poses);
  require(public_system.has_value() && public_system->gradient.size() == 4 &&
    public_system->hessian_blocks.size() == 16,
    "public linearization diagnostics");
  require(!abseil::evaluate(*prepared, std::span<const abseil::pose4>{}),
    "public pose count validation");
}
