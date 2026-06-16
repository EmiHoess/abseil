#include "internal.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iomanip>
#include <queue>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_set>
#include <utility>

namespace seil {
namespace {

class snapshot_hash {
public:
  void add_u64(std::uint64_t value) noexcept
  {
    for(unsigned shift = 0; shift < 64; shift += 8) {
      value_ ^= static_cast<std::uint8_t>(value >> shift);
      value_ *= 1099511628211ULL;
    }
  }

  void add_u32(std::uint32_t value) noexcept
  {
    for(unsigned shift = 0; shift < 32; shift += 8) {
      value_ ^= static_cast<std::uint8_t>(value >> shift);
      value_ *= 1099511628211ULL;
    }
  }

  void add_size(std::size_t value) noexcept
  {
    add_u64(static_cast<std::uint64_t>(value));
  }

  void add_bool(bool value) noexcept
  {
    add_u32(value ? 1U : 0U);
  }

  void add_double(double value) noexcept
  {
    add_u64(std::bit_cast<std::uint64_t>(value));
  }

  void add_float(float value) noexcept
  {
    add_u32(std::bit_cast<std::uint32_t>(value));
  }

  void add_pose(const pose4& value) noexcept
  {
    add_double(value.x);
    add_double(value.y);
    add_double(value.z);
    add_double(value.yaw);
  }

  std::string text() const
  {
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << value_;
    return output.str();
  }

private:
  std::uint64_t value_{1469598103934665603ULL};
};

operation_error preparation_error(std::string message)
{
  return {
    operation_code::invalid_problem,
    operation_stage::preparation,
    0,
    std::move(message)
  };
}

bool finite_matrix(const matrix4& value) noexcept
{
  return std::all_of(value.begin(), value.end(), [](double item) {
    return std::isfinite(item);
  });
}

bool valid_loss(const loss& value) noexcept
{
  return value.kind == loss_kind::none ||
    (value.scale > 0.0 && std::isfinite(value.scale));
}

void connect(
  std::vector<std::vector<std::size_t>>& adjacency,
  std::size_t left,
  std::size_t right
)
{
  adjacency[left].push_back(right);
  adjacency[right].push_back(left);
}

}

bool prepared_problem::empty() const noexcept
{
  return !storage_ || storage_->value.nodes.empty();
}

std::size_t prepared_problem::node_count() const noexcept
{
  return storage_ ? storage_->value.nodes.size() : 0U;
}

std::size_t prepared_problem::active_node_count() const noexcept
{
  return storage_ ? storage_->topology.active_count : 0U;
}

bool prepared_problem::has_unanchored_components() const noexcept
{
  return storage_ && storage_->topology.unanchored_components != 0;
}

const std::string& prepared_problem::hash() const noexcept
{
  static const std::string empty_hash;
  return storage_ ? storage_->hash : empty_hash;
}

prepared_problem::prepared_problem(
  std::shared_ptr<const detail::prepared_storage> storage
) : storage_(std::move(storage))
{
}

prepared_problem detail::prepared_problem_access::make(
  std::shared_ptr<const prepared_storage> storage
)
{
  return prepared_problem(std::move(storage));
}

const detail::prepared_storage* detail::prepared_problem_access::get(
  const prepared_problem& value
) noexcept
{
  return value.storage_.get();
}

operation_result<prepared_problem> prepare(problem value)
{
  if(value.nodes.empty())
    return std::unexpected(preparation_error("pose graph must contain at least one node"));

  auto storage = std::make_shared<detail::prepared_storage>();
  storage->value = std::move(value);
  problem& input = storage->value;

  std::unordered_set<std::uint64_t> node_ids;
  storage->topology.active_node.assign(input.nodes.size(), -1);
  for(std::size_t index = 0; index < input.nodes.size(); ++index) {
    node& current = input.nodes[index];
    if(!node_ids.insert(current.id).second)
      return std::unexpected(preparation_error("pose graph contains a duplicate node id"));
    if(!finite_pose(current.initial))
      return std::unexpected(preparation_error("pose graph contains a non-finite pose"));
    current.initial.yaw = wrap_yaw(current.initial.yaw);
    if(!current.constant) {
      storage->topology.active_node[index] =
        static_cast<std::int32_t>(storage->topology.active_count);
      ++storage->topology.active_count;
    }
  }

  const auto validate_factor = [&](std::size_t left, std::size_t right,
                                   const pose4& measured,
                                   const matrix4& information,
                                   const loss& robust_loss) -> operation_status {
    if(left >= input.nodes.size() || right >= input.nodes.size())
      return std::unexpected(preparation_error("factor references an unknown node"));
    if(!finite_pose(measured) || !finite_matrix(information))
      return std::unexpected(preparation_error("factor contains non-finite values"));
    if(!valid_loss(robust_loss))
      return std::unexpected(preparation_error("robust loss scale must be finite and positive"));
    return {};
  };

  for(relative_factor& factor : input.relative_factors) {
    factor.measured.yaw = wrap_yaw(factor.measured.yaw);
    operation_status status = validate_factor(
      factor.node_i,
      factor.node_j,
      factor.measured,
      factor.sqrt_information,
      factor.robust_loss
    );
    if(!status)
      return std::unexpected(std::move(status.error()));
  }

  for(absolute_factor& factor : input.absolute_factors) {
    factor.measured_relative.yaw = wrap_yaw(factor.measured_relative.yaw);
    operation_status status = validate_factor(
      factor.reference_node,
      factor.measured_node,
      factor.measured_relative,
      factor.sqrt_information,
      factor.robust_loss
    );
    if(!status)
      return std::unexpected(std::move(status.error()));
  }

  const registration_model& model = input.registration;
  if(!valid_loss(model.robust_loss) ||
     model.minimum_tsdf_weight < 0.0 ||
     !std::isfinite(model.minimum_tsdf_weight) ||
     !(model.maximum_confidence_weight > 0.0) ||
     !std::isfinite(model.maximum_confidence_weight) ||
     model.minimum_gradient_norm < 0.0 ||
     !std::isfinite(model.minimum_gradient_norm) ||
     model.maximum_absolute_residual_m < 0.0 ||
     !std::isfinite(model.maximum_absolute_residual_m))
    return std::unexpected(preparation_error("registration model is invalid"));

  for(const field_volume& volume : input.volumes) {
    if(volume.pose_node >= input.nodes.size() ||
       !(volume.voxel_size > 0.0) || !std::isfinite(volume.voxel_size))
      return std::unexpected(preparation_error("scalar-field volume is invalid"));
  }

  std::set<std::tuple<std::size_t, std::int32_t, std::int32_t, std::int32_t>> tsdf_keys;
  for(const tsdf_tile& tile : input.tsdf_tiles) {
    if(tile.volume >= input.volumes.size())
      return std::unexpected(preparation_error("TSDF tile references an unknown volume"));
    if(!input.volumes[tile.volume].has_tsdf)
      return std::unexpected(preparation_error("TSDF tile references a volume without TSDF data"));
    if(!tsdf_keys.emplace(tile.volume, tile.x, tile.y, tile.z).second)
      return std::unexpected(preparation_error("TSDF tile key is duplicated"));
  }

  std::set<std::tuple<std::size_t, std::int32_t, std::int32_t, std::int32_t>> esdf_keys;
  for(const esdf_tile& tile : input.esdf_tiles) {
    if(tile.volume >= input.volumes.size())
      return std::unexpected(preparation_error("ESDF tile references an unknown volume"));
    if(!input.volumes[tile.volume].has_esdf)
      return std::unexpected(preparation_error("ESDF tile references a volume without ESDF data"));
    if(!esdf_keys.emplace(tile.volume, tile.x, tile.y, tile.z).second)
      return std::unexpected(preparation_error("ESDF tile key is duplicated"));
  }

  for(std::size_t group_index = 0;
      group_index < input.registration_groups.size(); ++group_index) {
    const registration_group& group = input.registration_groups[group_index];
    if(group.node_i >= input.nodes.size() || group.node_j >= input.nodes.size() ||
       group.volume_i >= input.volumes.size() || group.volume_j >= input.volumes.size() ||
       input.volumes[group.volume_i].pose_node != group.node_i ||
       input.volumes[group.volume_j].pose_node != group.node_j ||
       group.sample_begin > input.registration_samples.size() ||
       group.sample_count > input.registration_samples.size() - group.sample_begin ||
       group.direction_weight < 0.0 || !std::isfinite(group.direction_weight))
      return std::unexpected(preparation_error("registration group is invalid"));

    const field_volume& target_volume = input.volumes[group.volume_j];
    if((group.field == field_kind::tsdf && !target_volume.has_tsdf) ||
       (group.field == field_kind::esdf && !target_volume.has_esdf))
      return std::unexpected(preparation_error("registration field is unavailable"));

    for(std::size_t offset = 0; offset < group.sample_count; ++offset) {
      const registration_sample& sample = input.registration_samples[group.sample_begin + offset];
      if(sample.group != group_index ||
         !finite_vector(sample.reference_local_point) ||
         !std::isfinite(sample.reference_distance) ||
         !std::isfinite(sample.reference_tsdf_weight))
        return std::unexpected(preparation_error("registration sample is invalid"));
    }
  }

  std::vector<std::vector<std::size_t>> adjacency(input.nodes.size());
  for(const relative_factor& factor : input.relative_factors)
    connect(adjacency, factor.node_i, factor.node_j);
  for(const absolute_factor& factor : input.absolute_factors)
    connect(adjacency, factor.reference_node, factor.measured_node);
  for(const registration_group& group : input.registration_groups)
    connect(adjacency, group.node_i, group.node_j);

  std::set<std::pair<std::int32_t, std::int32_t>> candidate_pairs;
  for(std::size_t index = 0; index < storage->topology.active_count; ++index) {
    const auto active = static_cast<std::int32_t>(index);
    candidate_pairs.emplace(active, active);
  }
  const auto add_candidate = [&](std::size_t left, std::size_t right) {
    const std::int32_t row = storage->topology.active_node[left];
    const std::int32_t column = storage->topology.active_node[right];
    if(row >= 0 && column >= 0)
      candidate_pairs.emplace(std::min(row, column), std::max(row, column));
  };
  for(const relative_factor& factor : input.relative_factors)
    add_candidate(factor.node_i, factor.node_j);
  for(const absolute_factor& factor : input.absolute_factors)
    add_candidate(factor.reference_node, factor.measured_node);
  for(const registration_group& group : input.registration_groups)
    add_candidate(group.node_i, group.node_j);
  storage->topology.sparse_candidates.reserve(candidate_pairs.size());
  for(const auto& [row, column] : candidate_pairs)
    storage->topology.sparse_candidates.push_back({row, column});

  std::vector<bool> visited(input.nodes.size());
  storage->topology.component_of_node.resize(input.nodes.size());
  for(std::size_t start = 0; start < input.nodes.size(); ++start) {
    if(visited[start])
      continue;
    bool has_active = false;
    bool has_constant = false;
    const std::size_t component_index = storage->topology.components.size();
    std::queue<std::size_t> pending;
    pending.push(start);
    visited[start] = true;
    while(!pending.empty()) {
      const std::size_t current = pending.front();
      pending.pop();
      storage->topology.component_of_node[current] = component_index;
      has_active = has_active || !input.nodes[current].constant;
      has_constant = has_constant || input.nodes[current].constant;
      for(const std::size_t next : adjacency[current]) {
        if(!visited[next]) {
          visited[next] = true;
          pending.push(next);
        }
      }
    }
    storage->topology.components.push_back({has_active, has_constant});
    if(has_active && !has_constant)
      ++storage->topology.unanchored_components;
  }

  snapshot_hash hash;
  hash.add_size(input.nodes.size());
  for(const node& current : input.nodes) {
    hash.add_u64(current.id);
    hash.add_pose(current.initial);
    hash.add_bool(current.constant);
  }

  hash.add_size(input.relative_factors.size());
  for(const relative_factor& factor : input.relative_factors) {
    hash.add_size(factor.node_i);
    hash.add_size(factor.node_j);
    hash.add_pose(factor.measured);
    for(const double item : factor.sqrt_information)
      hash.add_double(item);
    hash.add_u32(static_cast<std::uint32_t>(factor.robust_loss.kind));
    hash.add_double(factor.robust_loss.scale);
    hash.add_u64(factor.diagnostic_id);
  }

  hash.add_size(input.absolute_factors.size());
  for(const absolute_factor& factor : input.absolute_factors) {
    hash.add_size(factor.reference_node);
    hash.add_size(factor.measured_node);
    hash.add_pose(factor.measured_relative);
    for(const double item : factor.sqrt_information)
      hash.add_double(item);
    hash.add_u32(static_cast<std::uint32_t>(factor.robust_loss.kind));
    hash.add_double(factor.robust_loss.scale);
    hash.add_u64(factor.diagnostic_id);
  }

  hash.add_size(input.registration_groups.size());
  for(const registration_group& group : input.registration_groups) {
    hash.add_size(group.node_i);
    hash.add_size(group.node_j);
    hash.add_size(group.volume_i);
    hash.add_size(group.volume_j);
    hash.add_u32(static_cast<std::uint32_t>(group.field));
    hash.add_size(group.sample_begin);
    hash.add_size(group.sample_count);
    hash.add_double(group.direction_weight);
    hash.add_u64(group.diagnostic_id);
  }

  hash.add_size(input.registration_samples.size());
  for(const registration_sample& sample : input.registration_samples) {
    hash.add_size(sample.group);
    hash.add_double(sample.reference_local_point.x);
    hash.add_double(sample.reference_local_point.y);
    hash.add_double(sample.reference_local_point.z);
    hash.add_double(sample.reference_distance);
    hash.add_double(sample.reference_tsdf_weight);
    hash.add_u64(sample.ordinal);
  }

  hash.add_size(input.volumes.size());
  for(const field_volume& volume : input.volumes) {
    hash.add_size(volume.pose_node);
    hash.add_double(volume.voxel_size);
    hash.add_bool(volume.has_tsdf);
    hash.add_bool(volume.has_esdf);
  }

  hash.add_size(input.tsdf_tiles.size());
  for(const tsdf_tile& tile : input.tsdf_tiles) {
    hash.add_size(tile.volume);
    hash.add_u32(static_cast<std::uint32_t>(tile.x));
    hash.add_u32(static_cast<std::uint32_t>(tile.y));
    hash.add_u32(static_cast<std::uint32_t>(tile.z));
    for(const tsdf_voxel& voxel : tile.voxels) {
      hash.add_float(voxel.distance);
      hash.add_float(voxel.weight);
      hash.add_u32(voxel.flags);
    }
  }

  hash.add_size(input.esdf_tiles.size());
  for(const esdf_tile& tile : input.esdf_tiles) {
    hash.add_size(tile.volume);
    hash.add_u32(static_cast<std::uint32_t>(tile.x));
    hash.add_u32(static_cast<std::uint32_t>(tile.y));
    hash.add_u32(static_cast<std::uint32_t>(tile.z));
    for(const esdf_voxel& voxel : tile.voxels) {
      hash.add_float(voxel.distance);
      hash.add_u32(voxel.flags);
    }
  }

  hash.add_u32(static_cast<std::uint32_t>(input.registration.profile));
  hash.add_u32(static_cast<std::uint32_t>(input.registration.robust_loss.kind));
  hash.add_double(input.registration.robust_loss.scale);
  hash.add_double(input.registration.minimum_tsdf_weight);
  hash.add_double(input.registration.maximum_confidence_weight);
  hash.add_double(input.registration.minimum_gradient_norm);
  hash.add_double(input.registration.maximum_absolute_residual_m);
  hash.add_bool(input.registration.allow_nearest_fallback);
  storage->hash = hash.text();

  return detail::prepared_problem_access::make(std::move(storage));
}

}
