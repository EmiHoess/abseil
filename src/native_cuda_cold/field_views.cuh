#pragma once

#include "cuda_abi.h"

#include <cuda_runtime.h>

#include <cmath>

struct abseil_cuda_field_query {
  double distance;
  double weight;
  double gradient[3];
  bool gradient_valid;
};

__device__ inline bool abseil_read_field(
  const abseil_cuda_field_tile& tile,
  size_t local,
  double* distance,
  double* weight
)
{
  const size_t offset = local * tile.voxel_stride;
  const float value = *reinterpret_cast<const float*>(tile.distance + offset);
  const uint32_t flags = *reinterpret_cast<const uint32_t*>(tile.flags + offset);
  if((flags & 1U) == 0U || !isfinite(value))
    return false;
  float decoded = value;
  float confidence = 1.0F;
  if(tile.weight != nullptr) {
    confidence = *reinterpret_cast<const float*>(tile.weight + offset);
    if(!isfinite(confidence) || !(confidence > 0.0F))
      return false;
  } else if(tile.encoding == 1U) {
    if(value < 0.0F)
      return false;
    decoded = copysignf(fminf(tile.maximum_distance,
      tile.voxel_size * sqrtf(value)), (flags & 2U) != 0U ? -1.0F : 1.0F);
  } else {
    decoded = copysignf(fminf(tile.maximum_distance, fabsf(value)), value);
  }
  *distance = decoded;
  *weight = confidence;
  return true;
}

__device__ inline int abseil_floor_divide(int value, int divisor)
{
  int quotient = value / divisor;
  if(value % divisor < 0)
    --quotient;
  return quotient;
}

__device__ inline int abseil_floor_modulo(int value, int divisor)
{
  const int remainder = value % divisor;
  return remainder < 0 ? remainder + divisor : remainder;
}

__device__ inline bool abseil_lookup_field(
  uint32_t field,
  int32_t volume,
  int x,
  int y,
  int z,
  const abseil_cuda_field_tile* tsdf_views,
  size_t tsdf_view_count,
  const abseil_cuda_field_tile* esdf_views,
  size_t esdf_view_count,
  double* distance,
  double* weight
)
{
  const int tile_x = abseil_floor_divide(x, abseil_cuda_tile_edge);
  const int tile_y = abseil_floor_divide(y, abseil_cuda_tile_edge);
  const int tile_z = abseil_floor_divide(z, abseil_cuda_tile_edge);
  const size_t local = static_cast<size_t>(abseil_floor_modulo(x, 8)) +
    8U * (static_cast<size_t>(abseil_floor_modulo(y, 8)) +
    8U * static_cast<size_t>(abseil_floor_modulo(z, 8)));
  const abseil_cuda_field_tile* views = field == abseil_cuda_field_tsdf ?
    tsdf_views : esdf_views;
  const size_t view_count = field == abseil_cuda_field_tsdf ?
    tsdf_view_count : esdf_view_count;
  for(size_t index = 0; index < view_count; ++index) {
    const abseil_cuda_field_tile& tile = views[index];
    if(tile.volume_index == volume && tile.x == tile_x &&
      tile.y == tile_y && tile.z == tile_z)
      return abseil_read_field(tile, local, distance, weight);
  }
  return false;
}

__device__ inline bool abseil_query_trilinear(
  uint32_t field,
  int32_t volume,
  double voxel_size,
  double point_x,
  double point_y,
  double point_z,
  const abseil_cuda_field_tile* tsdf_views,
  size_t tsdf_view_count,
  const abseil_cuda_field_tile* esdf_views,
  size_t esdf_view_count,
  abseil_cuda_field_query* output
)
{
  const double coordinate_x = point_x / voxel_size - 0.5;
  const double coordinate_y = point_y / voxel_size - 0.5;
  const double coordinate_z = point_z / voxel_size - 0.5;
  const int base_x = static_cast<int>(floor(coordinate_x));
  const int base_y = static_cast<int>(floor(coordinate_y));
  const int base_z = static_cast<int>(floor(coordinate_z));
  const double tx = coordinate_x - base_x;
  const double ty = coordinate_y - base_y;
  const double tz = coordinate_z - base_z;
  double distances[2][2][2]{};
  double weights[2][2][2]{};
  for(int z = 0; z < 2; ++z)
    for(int y = 0; y < 2; ++y)
      for(int x = 0; x < 2; ++x)
        if(!abseil_lookup_field(field, volume, base_x + x, base_y + y,
          base_z + z, tsdf_views, tsdf_view_count, esdf_views, esdf_view_count,
          &distances[x][y][z], &weights[x][y][z]))
          return false;
  *output = {};
  for(int z = 0; z < 2; ++z)
    for(int y = 0; y < 2; ++y)
      for(int x = 0; x < 2; ++x) {
        const double basis = (x ? tx : 1.0 - tx) *
          (y ? ty : 1.0 - ty) * (z ? tz : 1.0 - tz);
        output->distance += basis * distances[x][y][z];
        output->weight += basis * weights[x][y][z];
      }
  for(int z = 0; z < 2; ++z)
    for(int y = 0; y < 2; ++y)
      output->gradient[0] += (distances[1][y][z] - distances[0][y][z]) *
        (y ? ty : 1.0 - ty) * (z ? tz : 1.0 - tz);
  for(int z = 0; z < 2; ++z)
    for(int x = 0; x < 2; ++x)
      output->gradient[1] += (distances[x][1][z] - distances[x][0][z]) *
        (x ? tx : 1.0 - tx) * (z ? tz : 1.0 - tz);
  for(int y = 0; y < 2; ++y)
    for(int x = 0; x < 2; ++x)
      output->gradient[2] += (distances[x][y][1] - distances[x][y][0]) *
        (x ? tx : 1.0 - tx) * (y ? ty : 1.0 - ty);
  for(double& component : output->gradient)
    component /= voxel_size;
  output->gradient_valid = true;
  return true;
}
