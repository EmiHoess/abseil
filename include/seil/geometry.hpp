#pragma once

#include <array>

namespace seil {

struct pose4 {
  double x{};
  double y{};
  double z{};
  double yaw{};
};

struct vector3 {
  double x{};
  double y{};
  double z{};
};

using matrix4 = std::array<double, 16>;

matrix4 identity_matrix4(double diagonal = 1.0) noexcept;
bool finite_pose(const pose4& value) noexcept;
bool finite_vector(const vector3& value) noexcept;
double wrap_yaw(double yaw) noexcept;
vector3 rotate_yaw(const vector3& value, double yaw) noexcept;
vector3 transform_point(const pose4& pose, const vector3& point) noexcept;
vector3 inverse_transform_point(const pose4& pose, const vector3& point) noexcept;
pose4 compose_pose(const pose4& left, const pose4& right) noexcept;
pose4 inverse_pose(const pose4& value) noexcept;
pose4 relative_pose(const pose4& from, const pose4& to) noexcept;

}
