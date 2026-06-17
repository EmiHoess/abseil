#include "internal.hpp"

#include <seil/objective.hpp>

#include <algorithm>
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
  const double dx = right.x - left.x;
  const double dy = right.y - left.y;
  const double px = cosine * dx + sine * dy;
  const double py = -sine * dx + cosine * dy;
  const double raw[4]{
    px - measured.x,
    py - measured.y,
    right.z - left.z - measured.z,
    wrap_yaw(right.yaw - left.yaw - measured.yaw)
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
