#include "internal.hpp"

#include <seil/objective.hpp>

#include <cmath>
#include <utility>

namespace seil::detail {

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

}

namespace seil {

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

}
