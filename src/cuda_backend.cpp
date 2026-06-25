#include "cuda_backend.hpp"
#include "internal.hpp"

#include <algorithm>
#include <array>
#include <limits>

namespace seil::detail {
namespace {

operation_error packing_error(std::string message)
{
  return {operation_code::invalid_problem, operation_stage::upload, 0,
    std::move(message)};
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
  auto left = cuda_packing::index(node_i);
  auto right = cuda_packing::index(node_j);
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

}

operation_result<std::int32_t> cuda_packing::index(std::size_t value)
{
  if(value > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
    return std::unexpected(packing_error("CUDA snapshot index exceeds int32 range"));
  return static_cast<std::int32_t>(value);
}

operation_result<std::uint32_t> cuda_packing::count(std::size_t value)
{
  if(value > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
    return std::unexpected(packing_error("CUDA snapshot count exceeds uint32 range"));
  return static_cast<std::uint32_t>(value);
}

operation_result<std::uint64_t> cuda_packing::add_bytes(
  std::uint64_t total,
  std::size_t count_value,
  std::size_t element_size
)
{
  if(count_value != 0 && element_size >
    std::numeric_limits<std::uint64_t>::max() / count_value)
    return std::unexpected(packing_error("CUDA snapshot byte count overflows"));
  const std::uint64_t addition = static_cast<std::uint64_t>(count_value) * element_size;
  if(total > std::numeric_limits<std::uint64_t>::max() - addition)
    return std::unexpected(packing_error("CUDA snapshot byte count overflows"));
  return total + addition;
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
    auto node_i = cuda_packing::index(group.node_i);
    auto node_j = cuda_packing::index(group.node_j);
    auto volume_i = cuda_packing::index(group.volume_i);
    auto volume_j = cuda_packing::index(group.volume_j);
    auto sample_begin = cuda_packing::count(group.sample_begin);
    auto sample_count = cuda_packing::count(group.sample_count);
    if(!node_i || !node_j || !volume_i || !volume_j ||
      !sample_begin || !sample_count)
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
      *sample_begin,
      *sample_count,
      group.direction_weight
    });
  }
  for(const auto& sample : storage->value.registration_samples) {
    auto group = cuda_packing::count(sample.group);
    if(!group)
      return std::unexpected(std::move(group.error()));
    output.registration_samples.push_back({
      *group,
      sample.reference_local_point.x,
      sample.reference_local_point.y,
      sample.reference_local_point.z,
      sample.reference_distance,
      sample.reference_tsdf_weight
    });
  }
  for(const auto& volume : storage->value.volumes) {
    auto node = cuda_packing::index(volume.pose_node);
    if(!node)
      return std::unexpected(std::move(node.error()));
    output.volumes.push_back({
      *node, volume.voxel_size,
      volume.has_tsdf ? 1U : 0U, volume.has_esdf ? 1U : 0U
    });
  }
  output.tsdf_tiles.reserve(storage->value.tsdf_tiles.size());
  for(const auto& tile : storage->value.tsdf_tiles) {
    seil_cuda_tsdf_tile packed{};
    auto volume = cuda_packing::index(tile.volume);
    if(!volume)
      return std::unexpected(std::move(volume.error()));
    packed.volume_index = *volume;
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
    auto volume = cuda_packing::index(tile.volume);
    if(!volume)
      return std::unexpected(std::move(volume.error()));
    packed.volume_index = *volume;
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

  const std::array ranges{
    std::pair{output.nodes.size(), sizeof(seil_cuda_node)},
    std::pair{output.relative_edges.size(), sizeof(seil_cuda_edge)},
    std::pair{output.absolute_edges.size(), sizeof(seil_cuda_edge)},
    std::pair{output.registration_groups.size(), sizeof(seil_cuda_registration_group)},
    std::pair{output.registration_samples.size(), sizeof(seil_cuda_registration_sample)},
    std::pair{output.volumes.size(), sizeof(seil_cuda_volume)},
    std::pair{output.tsdf_tiles.size(), sizeof(seil_cuda_tsdf_tile)},
    std::pair{output.esdf_tiles.size(), sizeof(seil_cuda_esdf_tile)},
    std::pair{output.sparse_slots.size(), sizeof(seil_cuda_sparse_slot)}
  };
  for(const auto& [count_value, element_size] : ranges) {
    auto total = cuda_packing::add_bytes(output.byte_count, count_value, element_size);
    if(!total)
      return std::unexpected(std::move(total.error()));
    output.byte_count = *total;
  }
  auto active_count = cuda_packing::index(storage->topology.active_count);
  if(!active_count)
    return std::unexpected(std::move(active_count.error()));
  output.view = {
    seil_cuda_abi_version_value,
    sizeof(seil_cuda_problem),
    output.nodes.data(), output.nodes.size(),
    *active_count,
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
  return true;
}

operation_result<solve_result> solve_native_cuda_cold(
  const prepared_problem& value,
  const common_options& common,
  const native_cuda_cold_options& options
)
{
  return solve_cuda_checked(value, common, options);
}

}
