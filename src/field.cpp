#include "internal.hpp"

#include <seil/field.hpp>

#include <cmath>

namespace seil {
namespace {

std::int32_t floor_divide(std::int32_t value, std::int32_t divisor) noexcept
{
  std::int32_t quotient = value / divisor;
  if(value % divisor < 0)
    --quotient;
  return quotient;
}

std::int32_t floor_modulo(std::int32_t value, std::int32_t divisor) noexcept
{
  const std::int32_t remainder = value % divisor;
  return remainder < 0 ? remainder + divisor : remainder;
}

bool lookup(
  const detail::prepared_storage& storage,
  field_kind field,
  std::size_t volume,
  std::int32_t x,
  std::int32_t y,
  std::int32_t z,
  field_sample& output
)
{
  const auto tile_x = floor_divide(x, static_cast<std::int32_t>(tile_edge));
  const auto tile_y = floor_divide(y, static_cast<std::int32_t>(tile_edge));
  const auto tile_z = floor_divide(z, static_cast<std::int32_t>(tile_edge));
  const std::size_t local = static_cast<std::size_t>(floor_modulo(x, 8)) +
    8U * (static_cast<std::size_t>(floor_modulo(y, 8)) +
    8U * static_cast<std::size_t>(floor_modulo(z, 8)));

  const detail::tile_key key{volume, tile_x, tile_y, tile_z};
  if(field == field_kind::tsdf) {
    const auto found = storage.tsdf_index.find(key);
    if(found == storage.tsdf_index.end())
      return false;
    const auto& voxel = storage.value.tsdf_tiles[found->second].voxels[local];
    if((voxel.flags & 1U) == 0U || !(voxel.weight > 0.0F) ||
      !std::isfinite(voxel.distance) || !std::isfinite(voxel.weight))
      return false;
    output = {voxel.distance, voxel.weight};
    return true;
  }
  const auto found = storage.esdf_index.find(key);
  if(found == storage.esdf_index.end())
    return false;
  const auto& voxel = storage.value.esdf_tiles[found->second].voxels[local];
  if((voxel.flags & 1U) == 0U || !std::isfinite(voxel.distance))
    return false;
  output = {voxel.distance, 1.0};
  return true;
}

bool nearest(
  const detail::prepared_storage& storage,
  field_kind field,
  std::size_t volume,
  double voxel_size,
  vector3 point,
  bool calculate_gradient,
  field_query& output
)
{
  const auto x = static_cast<std::int32_t>(std::floor(point.x / voxel_size));
  const auto y = static_cast<std::int32_t>(std::floor(point.y / voxel_size));
  const auto z = static_cast<std::int32_t>(std::floor(point.z / voxel_size));
  if(!lookup(storage, field, volume, x, y, z, output.sample))
    return false;
  if(!calculate_gradient)
    return true;

  field_sample x0, x1, y0, y1, z0, z1;
  if(lookup(storage, field, volume, x - 1, y, z, x0) &&
    lookup(storage, field, volume, x + 1, y, z, x1) &&
    lookup(storage, field, volume, x, y - 1, z, y0) &&
    lookup(storage, field, volume, x, y + 1, z, y1) &&
    lookup(storage, field, volume, x, y, z - 1, z0) &&
    lookup(storage, field, volume, x, y, z + 1, z1)) {
    const double scale = 0.5 / voxel_size;
    output.gradient = {
      (x1.distance - x0.distance) * scale,
      (y1.distance - y0.distance) * scale,
      (z1.distance - z0.distance) * scale
    };
    output.gradient_valid = true;
  }
  return true;
}

bool trilinear(
  const detail::prepared_storage& storage,
  field_kind field,
  std::size_t volume,
  double voxel_size,
  vector3 point,
  field_query& output
)
{
  const vector3 coordinate{
    point.x / voxel_size - 0.5,
    point.y / voxel_size - 0.5,
    point.z / voxel_size - 0.5
  };
  const auto bx = static_cast<std::int32_t>(std::floor(coordinate.x));
  const auto by = static_cast<std::int32_t>(std::floor(coordinate.y));
  const auto bz = static_cast<std::int32_t>(std::floor(coordinate.z));
  const double tx = coordinate.x - bx;
  const double ty = coordinate.y - by;
  const double tz = coordinate.z - bz;
  field_sample samples[2][2][2]{};
  for(int z = 0; z < 2; ++z)
    for(int y = 0; y < 2; ++y)
      for(int x = 0; x < 2; ++x)
        if(!lookup(storage, field, volume, bx + x, by + y, bz + z,
          samples[x][y][z]))
          return false;

  for(int z = 0; z < 2; ++z)
    for(int y = 0; y < 2; ++y)
      for(int x = 0; x < 2; ++x) {
        const double basis = (x ? tx : 1.0 - tx) *
          (y ? ty : 1.0 - ty) * (z ? tz : 1.0 - tz);
        output.sample.distance += basis * samples[x][y][z].distance;
        output.sample.weight += basis * samples[x][y][z].weight;
      }

  for(int z = 0; z < 2; ++z)
    for(int y = 0; y < 2; ++y)
      output.gradient.x +=
        (samples[1][y][z].distance - samples[0][y][z].distance) *
        (y ? ty : 1.0 - ty) * (z ? tz : 1.0 - tz);
  for(int z = 0; z < 2; ++z)
    for(int x = 0; x < 2; ++x)
      output.gradient.y +=
        (samples[x][1][z].distance - samples[x][0][z].distance) *
        (x ? tx : 1.0 - tx) * (z ? tz : 1.0 - tz);
  for(int y = 0; y < 2; ++y)
    for(int x = 0; x < 2; ++x)
      output.gradient.z +=
        (samples[x][y][1].distance - samples[x][y][0].distance) *
        (x ? tx : 1.0 - tx) * (y ? ty : 1.0 - ty);
  output.gradient.x /= voxel_size;
  output.gradient.y /= voxel_size;
  output.gradient.z /= voxel_size;
  output.gradient_valid = true;
  return true;
}

operation_error field_error(std::string message)
{
  return {operation_code::invalid_argument, operation_stage::validation, 0,
    std::move(message)};
}

}

operation_result<field_query> query_field(
  const prepared_problem& value,
  field_kind field,
  std::size_t volume,
  vector3 point,
  field_interpolation interpolation,
  bool calculate_gradient
)
{
  const auto* storage = detail::prepared_problem_access::get(value);
  if(storage == nullptr)
    return std::unexpected(field_error("problem is not prepared"));
  if(volume >= storage->value.volumes.size())
    return std::unexpected(field_error("field volume is out of range"));
  const auto& field_volume = storage->value.volumes[volume];
  if((field == field_kind::tsdf && !field_volume.has_tsdf) ||
    (field == field_kind::esdf && !field_volume.has_esdf))
    return std::unexpected(field_error("field is not available for volume"));

  field_query output;
  const bool found = interpolation == field_interpolation::nearest ?
    nearest(*storage, field, volume, field_volume.voxel_size, point,
      calculate_gradient, output) :
    trilinear(*storage, field, volume, field_volume.voxel_size, point, output);
  if(!found)
    return std::unexpected(field_error("field sample is unavailable"));
  return output;
}

}
