#pragma once

#include <seil/problem.hpp>

namespace seil {

struct robust_evaluation {
  double cost{};
  double weight{};
};

operation_result<robust_evaluation> evaluate_loss(
  const loss& robust_loss,
  double residual
);

}
