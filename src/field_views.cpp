#include "internal.hpp"

#include <seil/field_views.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <tuple>

namespace seil {
namespace {

operation_error field_view_error(std::string message)
{
  return {operation_code::invalid_problem, operation_stage::preparation, 0,
    std::move(message)};
}

}

operation_status validate_fields(
  const problem& value,
  const field_snapshot& fields
)
{
  if(fields.version != field_view_api_version || fields.identity == 0 ||
    fields.device < -1 || (!fields.tiles.empty() && fields.owners.empty()))
    return std::unexpected(field_view_error("field snapshot contract is invalid"));
  for(const auto& owner : fields.owners)
    if(!owner)
      return std::unexpected(field_view_error("field snapshot owner is absent"));

  std::set<std::tuple<field_kind, std::size_t, int, int, int>> keys;
  for(const auto& tile : fields.tiles) {
    const bool valid_kind = tile.kind == field_kind::tsdf ||
      tile.kind == field_kind::esdf;
    const bool valid_encoding = tile.encoding == field_encoding::signed_metres ||
      tile.encoding == field_encoding::squared_voxels;
    if(tile.volume >= value.volumes.size() || !valid_kind || !valid_encoding ||
      !keys.emplace(tile.kind, tile.volume, tile.x, tile.y, tile.z).second ||
      tile.content_generation == 0 || tile.allocation_generation == 0 ||
      tile.slot_generation == 0 || tile.distance == nullptr || tile.flags == nullptr ||
      (tile.kind == field_kind::tsdf && tile.weight == nullptr) ||
      tile.voxel_stride < sizeof(float) || tile.voxel_stride % alignof(float) != 0 ||
      tile.voxel_stride > static_cast<std::size_t>(PTRDIFF_MAX) / tile_voxels ||
      reinterpret_cast<std::uintptr_t>(tile.distance) % alignof(float) != 0 ||
      reinterpret_cast<std::uintptr_t>(tile.flags) % alignof(std::uint32_t) != 0 ||
      (tile.weight != nullptr && reinterpret_cast<std::uintptr_t>(tile.weight) %
        alignof(float) != 0) || !std::isfinite(tile.voxel_size) ||
      !(tile.voxel_size > 0.0F) || !std::isfinite(tile.maximum_distance) ||
      !(tile.maximum_distance > 0.0F) ||
      (tile.kind == field_kind::tsdf &&
        tile.encoding != field_encoding::signed_metres))
      return std::unexpected(field_view_error("borrowed field tile is invalid"));

    for(const auto& copied : value.tsdf_tiles)
      if(tile.kind == field_kind::tsdf && copied.volume == tile.volume &&
        copied.x == tile.x && copied.y == tile.y && copied.z == tile.z)
        return std::unexpected(field_view_error("copied and borrowed TSDF tiles overlap"));
    for(const auto& copied : value.esdf_tiles)
      if(tile.kind == field_kind::esdf && copied.volume == tile.volume &&
        copied.x == tile.x && copied.y == tile.y && copied.z == tile.z)
        return std::unexpected(field_view_error("copied and borrowed ESDF tiles overlap"));
  }
  return {};
}

bool read_field(
  const field_tile_view& tile,
  std::size_t local,
  double& distance,
  double& weight
) noexcept
{
  if(local >= tile_voxels)
    return false;
  float value = 0.0F;
  float confidence = 1.0F;
  std::uint32_t flags = 0;
  const std::size_t offset = local * tile.voxel_stride;
  std::memcpy(&value, tile.distance + offset, sizeof(value));
  std::memcpy(&flags, tile.flags + offset, sizeof(flags));
  if((flags & 1U) == 0U || !std::isfinite(value))
    return false;
  if(tile.kind == field_kind::tsdf) {
    std::memcpy(&confidence, tile.weight + offset, sizeof(confidence));
    if(!std::isfinite(confidence) || !(confidence > 0.0F))
      return false;
  } else if(tile.encoding == field_encoding::squared_voxels) {
    if(value < 0.0F)
      return false;
    value = std::copysign(std::min(tile.maximum_distance,
      tile.voxel_size * std::sqrt(value)), (flags & 2U) != 0U ? -1.0F : 1.0F);
  } else {
    value = std::copysign(std::min(tile.maximum_distance, std::abs(value)), value);
  }
  distance = value;
  weight = confidence;
  return true;
}

operation_result<prepared_problem> prepare_fields(
  problem value,
  std::shared_ptr<const field_snapshot> fields
)
{
  if(!fields)
    return std::unexpected(field_view_error("field snapshot is absent"));
  auto validation = validate_fields(value, *fields);
  if(!validation)
    return std::unexpected(std::move(validation.error()));
  auto prepared = prepare(std::move(value));
  if(!prepared)
    return std::unexpected(std::move(prepared.error()));
  const auto* source = detail::prepared_problem_access::get(*prepared);
  auto storage = std::make_shared<detail::prepared_storage>(*source);
  storage->fields = std::make_shared<const field_snapshot>(*fields);
  for(std::size_t index = 0; index < storage->fields->tiles.size(); ++index) {
    const auto& tile = storage->fields->tiles[index];
    auto& lookup = tile.kind == field_kind::tsdf ?
      storage->borrowed_tsdf_index : storage->borrowed_esdf_index;
    lookup.emplace(detail::tile_key{tile.volume, tile.x, tile.y, tile.z}, index);
    storage->hash += ":" + std::to_string(tile.content_generation) +
      ":" + std::to_string(tile.allocation_generation) +
      ":" + std::to_string(tile.slot_generation);
  }
  storage->hash += ":fields:" + std::to_string(fields->identity) +
    ":" + std::to_string(fields->device);
  for(std::uint64_t dependency : fields->dependencies)
    storage->hash += ":" + std::to_string(dependency);
  return detail::prepared_problem_access::make(std::move(storage));
}

bool cpu_fields_accessible(const prepared_problem& value) noexcept
{
  const auto* storage = detail::prepared_problem_access::get(value);
  return storage != nullptr && (!storage->fields || storage->fields->device == -1);
}

}
