#include <seil/objective.hpp>

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
  const seil::pose4& left,
  const seil::pose4& right,
  const seil::pose4& measured
)
{
  const auto row = seil::detail::evaluate_pose_factor(
    left, right, measured, seil::identity_matrix4(), {});
  return row.residual;
}

void adjust(seil::pose4& value, std::size_t axis, double amount)
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
  auto plain = seil::evaluate_loss({}, 2.0);
  require(plain.has_value(), "plain loss accepted");
  require(near(plain->cost, 2.0) && near(plain->weight, 1.0),
    "plain loss values");

  auto huber_edge = seil::evaluate_loss({seil::loss_kind::huber, 2.0}, 2.0);
  auto huber_tail = seil::evaluate_loss({seil::loss_kind::huber, 2.0}, -4.0);
  require(huber_edge.has_value() && near(huber_edge->cost, 2.0),
    "huber threshold cost");
  require(huber_tail.has_value() && near(huber_tail->cost, 6.0) &&
    near(huber_tail->weight, 0.5), "huber tail values");

  auto cauchy_zero = seil::evaluate_loss({seil::loss_kind::cauchy, 3.0}, 0.0);
  require(cauchy_zero.has_value() && near(cauchy_zero->cost, 0.0) &&
    near(cauchy_zero->weight, 1.0), "cauchy zero values");
  auto cauchy = seil::evaluate_loss({seil::loss_kind::cauchy, 2.0}, 2.0);
  require(cauchy.has_value() && near(cauchy->cost, 2.0 * std::log(2.0)) &&
    near(cauchy->weight, 0.5), "cauchy values");

  require(!seil::evaluate_loss({seil::loss_kind::huber, 0.0}, 1.0),
    "zero scale rejected");
  require(!seil::evaluate_loss({}, std::numeric_limits<double>::infinity()),
    "infinite residual rejected");

  const seil::pose4 left{1.0, -2.0, 0.5, 0.3};
  const seil::pose4 right{4.0, 1.0, -0.5, -0.2};
  const seil::pose4 measured{3.5, 1.8, -0.8, -0.4};
  const auto row = seil::detail::evaluate_pose_factor(
    left, right, measured, seil::identity_matrix4(),
    {seil::loss_kind::huber, 2.0});
  require(row.cost > 0.0 && row.weight > 0.0 && row.weight <= 1.0,
    "relative row robust accounting");

  constexpr double step = 1.0e-6;
  for(std::size_t column = 0; column < 4; ++column) {
    seil::pose4 moved = left;
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

  seil::problem graph;
  graph.nodes.push_back({1, left, true});
  graph.nodes.push_back({2, right, false});
  graph.relative_factors.push_back({0, 1, measured});
  auto prepared = seil::prepare(graph);
  require(prepared.has_value(), "relative graph preparation");
  const auto* storage = seil::detail::prepared_problem_access::get(*prepared);
  const std::array poses{left, right};
  const auto rows = seil::detail::evaluate_relative_factors(*storage, poses);
  require(rows.size() == 1 && rows[0].active_i == -1 && rows[0].active_j == 0,
    "fixed gauge activity");

  graph.relative_factors.clear();
  graph.absolute_factors.push_back({0, 1, measured,
    seil::identity_matrix4(2.0), {seil::loss_kind::cauchy, 3.0}, 91});
  prepared = seil::prepare(graph);
  require(prepared.has_value(), "absolute graph preparation");
  storage = seil::detail::prepared_problem_access::get(*prepared);
  const auto absolute_rows = seil::detail::evaluate_absolute_factors(*storage, poses);
  require(absolute_rows.size() == 1 && absolute_rows[0].active_i == -1 &&
    absolute_rows[0].active_j == 0, "absolute activity");
  const auto expected = seil::detail::evaluate_pose_factor(left, right, measured,
    seil::identity_matrix4(2.0), {seil::loss_kind::cauchy, 3.0});
  require(near(absolute_rows[0].cost, expected.cost) &&
    near(absolute_rows[0].weight, expected.weight),
    "absolute loss accounting");
  for(std::size_t index = 0; index < expected.residual.size(); ++index)
    require(near(absolute_rows[0].residual[index], expected.residual[index]),
      "absolute residual direction");

  graph.absolute_factors.clear();
  graph.volumes.push_back({0, 1.0, true, true});
  graph.volumes.push_back({1, 1.0, true, true});
  seil::tsdf_tile tsdf;
  seil::esdf_tile esdf;
  tsdf.volume = 1;
  esdf.volume = 1;
  for(std::size_t z = 0; z < seil::tile_edge; ++z)
    for(std::size_t y = 0; y < seil::tile_edge; ++y)
      for(std::size_t x = 0; x < seil::tile_edge; ++x) {
        const std::size_t index = x + 8U * (y + 8U * z);
        const float distance = static_cast<float>(x);
        tsdf.voxels[index] = {distance, 4.0F, 1U};
        esdf.voxels[index] = {distance, 1U};
      }
  graph.tsdf_tiles.push_back(tsdf);
  graph.esdf_tiles.push_back(esdf);
  graph.registration_samples.push_back({0, {2.5, 2.5, 2.5}, 2.0, 3.0, 1});
  graph.registration_groups.push_back({0, 1, 0, 1, seil::field_kind::esdf,
    0, 1, 1.0, 7});
  graph.nodes[0].initial = {};
  graph.nodes[1].initial = {};
  graph.registration.profile = seil::registration_profile::uniform_trilinear;
  prepared = seil::prepare(graph);
  require(prepared.has_value(), "registration graph preparation");
  storage = seil::detail::prepared_problem_access::get(*prepared);
  const std::array registration_poses{graph.nodes[0].initial, graph.nodes[1].initial};
  seil::registration_statistics statistics;
  double confidence = 0.0;
  const auto registration_rows = seil::detail::evaluate_registration_group(
    *storage, storage->value.registration_groups[0], registration_poses,
    confidence, &statistics);
  require(registration_rows.size() == 1 && registration_rows[0].rejection == 0,
    "registration row accepted");
  require(near(registration_rows[0].residual, 0.0) && near(confidence, 1.0) &&
    statistics.accepted_rows == 1, "registration row values");

  graph.registration.profile = seil::registration_profile::confidence_weighted;
  graph.registration.maximum_confidence_weight = 2.5;
  prepared = seil::prepare(graph);
  require(prepared.has_value(), "confidence graph preparation");
  storage = seil::detail::prepared_problem_access::get(*prepared);
  statistics = {};
  const auto confidence_rows = seil::detail::evaluate_registration_group(
    *storage, storage->value.registration_groups[0], registration_poses,
    confidence, &statistics);
  require(confidence_rows[0].rejection == 0 && near(confidence, 2.5),
    "confidence registration row");

  const seil::pose4 source_pose{1.0, 2.0, 3.0, 0.2};
  const seil::pose4 target_pose{-1.0, 0.5, 1.0, -0.4};
  const seil::vector3 local_point{2.0, -3.0, 4.0};
  const auto world_point = seil::transform_point(source_pose, local_point);
  const auto direct_point = seil::transform_point_between(
    source_pose, target_pose, local_point);
  const auto composed_point = seil::inverse_transform_point(target_pose, world_point);
  require(near(direct_point.x, composed_point.x) &&
    near(direct_point.y, composed_point.y) && near(direct_point.z, composed_point.z),
    "shared frame transformation");
  const auto derivative = seil::rotate_yaw_derivative(local_point, source_pose.yaw);
  const auto plus_rotation = seil::rotate_yaw(local_point, source_pose.yaw + step);
  const auto minus_rotation = seil::rotate_yaw(local_point, source_pose.yaw - step);
  require(std::abs(derivative.x - (plus_rotation.x - minus_rotation.x) /
    (2.0 * step)) < 1.0e-6, "shared yaw derivative");

  auto assembled = seil::detail::assemble_objective(
    *storage, registration_poses, true);
  require(assembled.has_value(), "objective assembly");
  require(assembled->cost.registration >= 0.0 &&
    assembled->registration.accepted_rows == 1, "objective split diagnostics");
  require(assembled->gradient.size() == 4 &&
    assembled->dense_hessian.size() == 16, "objective derivative dimensions");
  require(!seil::detail::assemble_objective(*storage,
    std::span<const seil::pose4>{}, false), "objective pose count validation");

  auto public_cost = seil::evaluate(*prepared, registration_poses);
  require(public_cost.has_value() && public_cost->registration.accepted_rows == 1,
    "public objective diagnostics");
  auto public_system = seil::linearize(*prepared, registration_poses);
  require(public_system.has_value() && public_system->gradient.size() == 4 &&
    public_system->hessian_blocks.size() == 16,
    "public linearization diagnostics");
  require(!seil::evaluate(*prepared, std::span<const seil::pose4>{}),
    "public pose count validation");
}
