#pragma once

#include <abseil/problem.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace abseil {

inline constexpr std::uint32_t field_view_api_version = 1;
inline constexpr std::uint32_t cuda_registration_coordinate_bits = 64;

enum class field_encoding : std::uint32_t {
  signed_metres,
  squared_voxels
};

struct field_tile_view {
  std::size_t volume{};
  std::int32_t x{};
  std::int32_t y{};
  std::int32_t z{};
  field_kind kind{field_kind::tsdf};
  const std::byte* distance{};
  const std::byte* weight{};
  const std::byte* flags{};
  std::size_t voxel_stride{};
  field_encoding encoding{field_encoding::signed_metres};
  float voxel_size{1.0F};
  float maximum_distance{};
  std::uint64_t content_generation{};
  std::uint64_t allocation_generation{};
  std::uint64_t slot_generation{};
};

struct field_snapshot {
  std::uint32_t version{field_view_api_version};
  std::uint64_t identity{};
  int device{-1};
  std::vector<field_tile_view> tiles;
  std::vector<std::shared_ptr<const void>> owners;
  std::vector<std::uint64_t> dependencies;
};

operation_status validate_fields(
  const problem& value,
  const field_snapshot& fields
);

operation_result<prepared_problem> prepare_fields(
  problem value,
  std::shared_ptr<const field_snapshot> fields
);

bool cpu_fields_accessible(const prepared_problem& value) noexcept;

bool read_field(
  const field_tile_view& tile,
  std::size_t local,
  double& distance,
  double& weight
) noexcept;

}
