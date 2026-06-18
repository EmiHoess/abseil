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
}
