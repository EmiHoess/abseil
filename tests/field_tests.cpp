#include <abseil/field.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

void require(bool condition, const char* message)
{
  if(!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

abseil::prepared_problem field_problem()
{
  abseil::problem value;
  value.nodes.push_back({1, {}, true});
  value.volumes.push_back({0, 1.0, true, true});
  abseil::tsdf_tile tsdf;
  abseil::esdf_tile esdf;
  tsdf.volume = 0;
  esdf.volume = 0;
  for(std::size_t z = 0; z < abseil::tile_edge; ++z)
    for(std::size_t y = 0; y < abseil::tile_edge; ++y)
      for(std::size_t x = 0; x < abseil::tile_edge; ++x) {
        const std::size_t index = x + 8U * (y + 8U * z);
        const float distance = static_cast<float>(x + 2U * y + 3U * z);
        tsdf.voxels[index] = {distance, 2.0F, 1U};
        esdf.voxels[index] = {distance, 1U};
      }
  tsdf.voxels[6U + 8U * (6U + 8U * 6U)].flags = 0;
  esdf.voxels[6U + 8U * (6U + 8U * 6U)].flags = 0;
  value.tsdf_tiles.push_back(tsdf);
  value.esdf_tiles.push_back(esdf);
  tsdf.x = -1;
  esdf.x = -1;
  for(auto& voxel : tsdf.voxels)
    voxel.distance += 100.0F;
  for(auto& voxel : esdf.voxels)
    voxel.distance += 100.0F;
  value.tsdf_tiles.push_back(tsdf);
  value.esdf_tiles.push_back(esdf);
  auto prepared = abseil::prepare(std::move(value));
  require(prepared.has_value(), "field problem preparation");
  return std::move(*prepared);
}

}

int main()
{
  const auto problem = field_problem();
  auto nearest = abseil::query_field(problem, abseil::field_kind::tsdf, 0,
    {2.2, 3.4, 4.1}, abseil::field_interpolation::nearest);
  require(nearest.has_value(), "nearest sample exists");
  require(std::abs(nearest->sample.distance - 20.0) < 1.0e-12,
    "nearest distance");
  require(nearest->sample.weight == 2.0, "tsdf weight");
  require(nearest->gradient_valid, "nearest gradient exists");
  require(std::abs(nearest->gradient.x - 1.0) < 1.0e-12 &&
    std::abs(nearest->gradient.y - 2.0) < 1.0e-12 &&
    std::abs(nearest->gradient.z - 3.0) < 1.0e-12,
    "nearest gradient");

  auto linear = abseil::query_field(problem, abseil::field_kind::esdf, 0,
    {2.5, 3.5, 4.5}, abseil::field_interpolation::trilinear);
  require(linear.has_value(), "trilinear sample exists");
  require(std::abs(linear->sample.distance - 20.0) < 1.0e-12,
    "trilinear distance");
  require(linear->gradient_valid, "trilinear gradient exists");

  auto negative = abseil::query_field(problem, abseil::field_kind::esdf, 0,
    {-0.2, 2.2, 2.2}, abseil::field_interpolation::nearest, false);
  require(negative.has_value(), "negative tile sample exists");
  require(negative->sample.distance > 100.0, "negative tile selected");

  auto boundary = abseil::query_field(problem, abseil::field_kind::esdf, 0,
    {0.0, 0.0, 0.0}, abseil::field_interpolation::trilinear);
  require(!boundary.has_value(), "missing boundary rejected");
  auto bad_volume = abseil::query_field(problem, abseil::field_kind::esdf, 4,
    {}, abseil::field_interpolation::nearest);
  require(!bad_volume.has_value(), "invalid volume rejected");
  auto unobserved = abseil::query_field(problem, abseil::field_kind::tsdf, 0,
    {6.2, 6.2, 6.2}, abseil::field_interpolation::nearest, false);
  require(!unobserved.has_value(), "unobserved voxel rejected");
}
