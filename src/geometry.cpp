#include <seil/geometry.hpp>

#include <cmath>
#include <numbers>

namespace seil {

matrix4 identity_matrix4(double diagonal) noexcept
{
  matrix4 result{};
  result[0] = diagonal;
  result[5] = diagonal;
  result[10] = diagonal;
  result[15] = diagonal;
  return result;
}

bool finite_pose(const pose4& value) noexcept
{
  return std::isfinite(value.x) && std::isfinite(value.y) &&
    std::isfinite(value.z) && std::isfinite(value.yaw);
}

bool finite_vector(const vector3& value) noexcept
{
  return std::isfinite(value.x) && std::isfinite(value.y) &&
    std::isfinite(value.z);
}

double wrap_yaw(double yaw) noexcept
{
  if(!std::isfinite(yaw))
    return yaw;

  constexpr double period = 2.0 * std::numbers::pi_v<double>;
  double result = std::remainder(yaw, period);
  if(result <= -std::numbers::pi_v<double>)
    result += period;
  return result;
}

vector3 rotate_yaw(const vector3& value, double yaw) noexcept
{
  const double cosine = std::cos(yaw);
  const double sine = std::sin(yaw);
  return {
    cosine * value.x - sine * value.y,
    sine * value.x + cosine * value.y,
    value.z
  };
}

vector3 rotate_yaw_derivative(const vector3& value, double yaw) noexcept
{
  const double cosine = std::cos(yaw);
  const double sine = std::sin(yaw);
  return {
    -sine * value.x - cosine * value.y,
    cosine * value.x - sine * value.y,
    0.0
  };
}

vector3 transform_point(const pose4& pose, const vector3& point) noexcept
{
  vector3 result = rotate_yaw(point, pose.yaw);
  result.x += pose.x;
  result.y += pose.y;
  result.z += pose.z;
  return result;
}

vector3 inverse_transform_point(const pose4& pose, const vector3& point) noexcept
{
  const vector3 translated{
    point.x - pose.x,
    point.y - pose.y,
    point.z - pose.z
  };
  return rotate_yaw(translated, -pose.yaw);
}

vector3 transform_point_between(
  const pose4& from,
  const pose4& to,
  const vector3& point
) noexcept
{
  return inverse_transform_point(to, transform_point(from, point));
}

pose4 compose_pose(const pose4& left, const pose4& right) noexcept
{
  const vector3 translation = rotate_yaw({right.x, right.y, right.z}, left.yaw);
  return {
    left.x + translation.x,
    left.y + translation.y,
    left.z + translation.z,
    wrap_yaw(left.yaw + right.yaw)
  };
}

pose4 inverse_pose(const pose4& value) noexcept
{
  const vector3 translation = rotate_yaw({-value.x, -value.y, -value.z}, -value.yaw);
  return {translation.x, translation.y, translation.z, wrap_yaw(-value.yaw)};
}

pose4 relative_pose(const pose4& from, const pose4& to) noexcept
{
  return compose_pose(inverse_pose(from), to);
}

}
