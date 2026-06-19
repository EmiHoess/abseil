#pragma once

#include <seil/problem.hpp>
#include <seil/solver.hpp>

#include <span>

namespace seil {

struct robust_evaluation {
  double cost{};
  double weight{};
};

struct objective_snapshot {
  cost_breakdown cost{};
  registration_statistics registration{};
};

operation_result<robust_evaluation> evaluate_loss(
  const loss& robust_loss,
  double residual
);

operation_result<objective_snapshot> evaluate(
  const prepared_problem& value,
  std::span<const pose4> poses
);

operation_result<linearization_snapshot> linearize(
  const prepared_problem& value,
  std::span<const pose4> poses
);

}
