#include "internal.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <set>
#include <tuple>
#include <unordered_set>
#include <utility>

namespace seil {
namespace {

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
  storage->active_index.assign(input.nodes.size(), -1);
  for(std::size_t index = 0; index < input.nodes.size(); ++index) {
    node& current = input.nodes[index];
    if(!node_ids.insert(current.id).second)
      return std::unexpected(preparation_error("pose graph contains a duplicate node id"));
    if(!finite_pose(current.initial))
      return std::unexpected(preparation_error("pose graph contains a non-finite pose"));
    current.initial.yaw = wrap_yaw(current.initial.yaw);
    if(!current.constant) {
      storage->active_index[index] = static_cast<std::int32_t>(storage->active_count);
      ++storage->active_count;
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

  std::vector<bool> visited(input.nodes.size());
  for(std::size_t start = 0; start < input.nodes.size(); ++start) {
    if(visited[start])
      continue;
    bool has_active = false;
    bool has_constant = false;
    std::queue<std::size_t> pending;
    pending.push(start);
    visited[start] = true;
    while(!pending.empty()) {
      const std::size_t current = pending.front();
      pending.pop();
      has_active = has_active || !input.nodes[current].constant;
      has_constant = has_constant || input.nodes[current].constant;
      for(const std::size_t next : adjacency[current]) {
        if(!visited[next]) {
          visited[next] = true;
          pending.push(next);
        }
      }
    }
    if(has_active && !has_constant)
      ++storage->unanchored_components;
  }

  return detail::prepared_problem_access::make(std::move(storage));
}

}
