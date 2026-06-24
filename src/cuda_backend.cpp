#include "cuda_backend.hpp"
#include "internal.hpp"

#include <algorithm>
#include <limits>

namespace seil::detail {
namespace {

operation_error packing_error(std::string message)
{
  return {operation_code::invalid_problem, operation_stage::upload, 0,
    std::move(message)};
}

operation_result<std::int32_t> narrow_index(std::size_t value)
{
  if(value > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
    return std::unexpected(packing_error("CUDA snapshot index exceeds int32 range"));
  return static_cast<std::int32_t>(value);
}

seil_cuda_loss pack_loss(const loss& value)
{
  return {static_cast<std::uint32_t>(value.kind), value.scale};
}

seil_cuda_pose4 pack_pose(const pose4& value)
{
  return {value.x, value.y, value.z, value.yaw};
}

operation_result<seil_cuda_edge> pack_edge(
  const topology_index& topology,
  std::size_t node_i,
  std::size_t node_j,
  const pose4& measured,
  const matrix4& sqrt_information,
  const loss& robust_loss
)
{
  auto left = narrow_index(node_i);
  auto right = narrow_index(node_j);
  if(!left)
    return std::unexpected(std::move(left.error()));
  if(!right)
    return std::unexpected(std::move(right.error()));
  const std::int32_t active_i = topology.active_node[node_i];
  const std::int32_t active_j = topology.active_node[node_j];
  const bool transpose = active_i > active_j;
  seil_cuda_edge output{};
  output.node_i = *left;
  output.node_j = *right;
  output.slot_ii = sparse_slot(topology, active_i, active_i);
  output.slot_ij = sparse_slot(topology, active_i, active_j);
  output.slot_jj = sparse_slot(topology, active_j, active_j);
  output.transpose_ij = transpose ? 1U : 0U;
  output.measured = pack_pose(measured);
  std::copy(sqrt_information.begin(), sqrt_information.end(),
    output.sqrt_information);
  output.robust_loss = pack_loss(robust_loss);
  return output;
}

template<class value_type>
std::uint64_t bytes(const std::vector<value_type>& values)
{
  return static_cast<std::uint64_t>(values.size()) * sizeof(value_type);
}

}

operation_result<packed_cuda_problem> pack_cuda_problem(
  const prepared_problem& value
)
{
  const auto* storage = prepared_problem_access::get(value);
  if(storage == nullptr)
    return std::unexpected(packing_error("problem is not prepared"));
  packed_cuda_problem output;
  output.nodes.reserve(storage->value.nodes.size());
  for(std::size_t index = 0; index < storage->value.nodes.size(); ++index)
    output.nodes.push_back({pack_pose(storage->value.nodes[index].initial),
      storage->topology.active_node[index]});
  output.relative_edges.reserve(storage->value.relative_factors.size());
  for(const auto& factor : storage->value.relative_factors) {
    auto edge = pack_edge(storage->topology, factor.node_i, factor.node_j,
      factor.measured, factor.sqrt_information, factor.robust_loss);
    if(!edge)
      return std::unexpected(std::move(edge.error()));
    output.relative_edges.push_back(*edge);
  }
  output.absolute_edges.reserve(storage->value.absolute_factors.size());
  for(const auto& factor : storage->value.absolute_factors) {
    auto edge = pack_edge(storage->topology, factor.reference_node,
      factor.measured_node, factor.measured_relative,
      factor.sqrt_information, factor.robust_loss);
    if(!edge)
      return std::unexpected(std::move(edge.error()));
    output.absolute_edges.push_back(*edge);
  }
  for(const auto& group : storage->value.registration_groups) {
    auto node_i = narrow_index(group.node_i);
    auto node_j = narrow_index(group.node_j);
    auto volume_i = narrow_index(group.volume_i);
    auto volume_j = narrow_index(group.volume_j);
    if(!node_i || !node_j || !volume_i || !volume_j)
      return std::unexpected(packing_error("CUDA registration index exceeds int32 range"));
    const auto active_i = storage->topology.active_node[group.node_i];
    const auto active_j = storage->topology.active_node[group.node_j];
    output.registration_groups.push_back({
      *node_i, *node_j, *volume_i, *volume_j,
      sparse_slot(storage->topology, active_i, active_i),
      sparse_slot(storage->topology, active_i, active_j),
      sparse_slot(storage->topology, active_j, active_j),
      active_i > active_j ? 1U : 0U,
      static_cast<std::uint32_t>(group.field),
      static_cast<std::uint32_t>(group.sample_begin),
      static_cast<std::uint32_t>(group.sample_count),
      group.direction_weight
    });
  }
  for(const auto& sample : storage->value.registration_samples)
    output.registration_samples.push_back({
      static_cast<std::uint32_t>(sample.group),
      sample.reference_local_point.x,
      sample.reference_local_point.y,
      sample.reference_local_point.z,
      sample.reference_distance,
      sample.reference_tsdf_weight
    });
  for(const auto& volume : storage->value.volumes)
    output.volumes.push_back({
      *narrow_index(volume.pose_node), volume.voxel_size,
      volume.has_tsdf ? 1U : 0U, volume.has_esdf ? 1U : 0U
    });
  output.tsdf_tiles.reserve(storage->value.tsdf_tiles.size());
  for(const auto& tile : storage->value.tsdf_tiles) {
    seil_cuda_tsdf_tile packed{};
    packed.volume_index = *narrow_index(tile.volume);
    packed.x = tile.x;
    packed.y = tile.y;
    packed.z = tile.z;
    for(std::size_t index = 0; index < tile.voxels.size(); ++index)
      packed.voxels[index] = {tile.voxels[index].distance,
        tile.voxels[index].weight, tile.voxels[index].flags};
    output.tsdf_tiles.push_back(packed);
  }
  output.esdf_tiles.reserve(storage->value.esdf_tiles.size());
  for(const auto& tile : storage->value.esdf_tiles) {
    seil_cuda_esdf_tile packed{};
    packed.volume_index = *narrow_index(tile.volume);
    packed.x = tile.x;
    packed.y = tile.y;
    packed.z = tile.z;
    for(std::size_t index = 0; index < tile.voxels.size(); ++index)
      packed.voxels[index] = {tile.voxels[index].distance,
        tile.voxels[index].flags};
    output.esdf_tiles.push_back(packed);
  }
  for(const auto& slot : storage->topology.sparse_candidates)
    output.sparse_slots.push_back({slot.row, slot.column});

  output.byte_count = bytes(output.nodes) + bytes(output.relative_edges) +
    bytes(output.absolute_edges) + bytes(output.registration_groups) +
    bytes(output.registration_samples) + bytes(output.volumes) +
    bytes(output.tsdf_tiles) + bytes(output.esdf_tiles) + bytes(output.sparse_slots);
  output.view = {
    seil_cuda_abi_version_value,
    sizeof(seil_cuda_problem),
    output.nodes.data(), output.nodes.size(),
    static_cast<std::int32_t>(storage->topology.active_count),
    output.relative_edges.data(), output.relative_edges.size(),
    output.absolute_edges.data(), output.absolute_edges.size(),
    output.registration_groups.data(), output.registration_groups.size(),
    output.registration_samples.data(), output.registration_samples.size(),
    output.volumes.data(), output.volumes.size(),
    output.tsdf_tiles.data(), output.tsdf_tiles.size(),
    output.esdf_tiles.data(), output.esdf_tiles.size(),
    output.sparse_slots.data(), output.sparse_slots.size(),
    storage->hash.c_str()
  };
  return output;
}

bool cuda_backend_available() noexcept
{
  return false;
}

operation_result<solve_result> solve_native_cuda_cold(
  const prepared_problem&,
  const common_options&,
  const native_cuda_cold_options&
)
{
  return std::unexpected(operation_error{operation_code::backend_unavailable,
    operation_stage::dispatch, 0,
    "native_cuda_cold backend is not exposed by this build stage"});
}

}
