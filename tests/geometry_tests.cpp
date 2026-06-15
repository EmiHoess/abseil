#include <seil/geometry.hpp>

#include <cmath>
#include <numbers>

namespace {

bool close(double left, double right, double tolerance = 1.0e-12)
{
  return std::abs(left - right) <= tolerance;
}

}

int main()
{
  using namespace seil;

  const matrix4 identity = identity_matrix4();
  if(!close(identity[0], 1.0) || !close(identity[5], 1.0) ||
     !close(identity[10], 1.0) || !close(identity[15], 1.0))
    return 1;

  if(!close(wrap_yaw(3.0 * std::numbers::pi_v<double>), std::numbers::pi_v<double>))
    return 2;

  const pose4 left{1.0, 2.0, -1.0, std::numbers::pi_v<double> / 2.0};
  const pose4 right{2.0, 0.0, 3.0, -0.25};
  const pose4 combined = compose_pose(left, right);
  const pose4 recovered = relative_pose(left, combined);
  if(!close(recovered.x, right.x) || !close(recovered.y, right.y) ||
     !close(recovered.z, right.z) || !close(recovered.yaw, right.yaw))
    return 3;

  const vector3 point{0.5, -2.0, 4.0};
  const vector3 transformed = transform_point(left, point);
  const vector3 restored = inverse_transform_point(left, transformed);
  if(!close(restored.x, point.x) || !close(restored.y, point.y) ||
     !close(restored.z, point.z))
    return 4;

  if(!finite_pose(left) || !finite_vector(point))
    return 5;

  return 0;
}
