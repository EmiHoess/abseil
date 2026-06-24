#pragma once

#include "cuda_abi.h"

#include <cuda_runtime.h>

#include <cmath>

struct seil_cuda_field_query {
  double distance;
  double weight;
  double gradient[3];
  bool gradient_valid;
};

__device__ inline int seil_floor_divide(int value, int divisor)
{
  int quotient = value / divisor;
  if(value % divisor < 0)
    --quotient;
  return quotient;
}

__device__ inline int seil_floor_modulo(int value, int divisor)
{
  const int remainder = value % divisor;
  return remainder < 0 ? remainder + divisor : remainder;
}

__device__ inline bool seil_lookup_field(
  uint32_t field,
  int32_t volume,
  int x,
  int y,
  int z,
  const seil_cuda_tsdf_tile* tsdf_tiles,
  size_t tsdf_count,
  const seil_cuda_esdf_tile* esdf_tiles,
  size_t esdf_count,
  double* distance,
  double* weight
)
{
  const int tile_x = seil_floor_divide(x, seil_cuda_tile_edge);
  const int tile_y = seil_floor_divide(y, seil_cuda_tile_edge);
  const int tile_z = seil_floor_divide(z, seil_cuda_tile_edge);
  const size_t local = static_cast<size_t>(seil_floor_modulo(x, 8)) +
    8U * (static_cast<size_t>(seil_floor_modulo(y, 8)) +
    8U * static_cast<size_t>(seil_floor_modulo(z, 8)));
  if(field == seil_cuda_field_tsdf) {
    for(size_t index = 0; index < tsdf_count; ++index) {
      const seil_cuda_tsdf_tile& tile = tsdf_tiles[index];
      if(tile.volume_index != volume || tile.x != tile_x ||
        tile.y != tile_y || tile.z != tile_z)
        continue;
      const seil_cuda_tsdf_voxel& voxel = tile.voxels[local];
      if((voxel.flags & 1U) == 0U || !(voxel.weight > 0.0F) ||
        !isfinite(voxel.distance) || !isfinite(voxel.weight))
        return false;
      *distance = voxel.distance;
      *weight = voxel.weight;
      return true;
    }
    return false;
  }
  for(size_t index = 0; index < esdf_count; ++index) {
    const seil_cuda_esdf_tile& tile = esdf_tiles[index];
    if(tile.volume_index != volume || tile.x != tile_x ||
      tile.y != tile_y || tile.z != tile_z)
      continue;
    const seil_cuda_esdf_voxel& voxel = tile.voxels[local];
    if((voxel.flags & 1U) == 0U || !isfinite(voxel.distance))
      return false;
    *distance = voxel.distance;
    *weight = 1.0;
    return true;
  }
  return false;
}

__device__ inline bool seil_query_trilinear(
  uint32_t field,
  int32_t volume,
  double voxel_size,
  double point_x,
  double point_y,
  double point_z,
  const seil_cuda_tsdf_tile* tsdf_tiles,
  size_t tsdf_count,
  const seil_cuda_esdf_tile* esdf_tiles,
  size_t esdf_count,
  seil_cuda_field_query* output
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
        if(!seil_lookup_field(field, volume, base_x + x, base_y + y,
          base_z + z, tsdf_tiles, tsdf_count, esdf_tiles, esdf_count,
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
