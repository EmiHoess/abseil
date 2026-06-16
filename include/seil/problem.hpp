#pragma once

#include <seil/error.hpp>
#include <seil/geometry.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace seil {

inline constexpr std::size_t tile_edge = 8;
inline constexpr std::size_t tile_voxels = tile_edge * tile_edge * tile_edge;

enum class loss_kind : std::uint32_t {
  none,
  huber,
  cauchy
};

struct loss {
  loss_kind kind{loss_kind::none};
  double scale{1.0};
};

struct node {
  std::uint64_t id{};
  pose4 initial{};
  bool constant{};
};

struct relative_factor {
  std::size_t node_i{};
  std::size_t node_j{};
  pose4 measured{};
  matrix4 sqrt_information{identity_matrix4()};
  loss robust_loss{};
  std::uint64_t diagnostic_id{};
};

struct absolute_factor {
  std::size_t reference_node{};
  std::size_t measured_node{};
  pose4 measured_relative{};
  matrix4 sqrt_information{identity_matrix4()};
  loss robust_loss{};
  std::uint64_t diagnostic_id{};
};

enum class field_kind : std::uint32_t {
  tsdf,
  esdf
};

enum class registration_profile : std::uint32_t {
  confidence_weighted,
  uniform_trilinear
};

struct tsdf_voxel {
  float distance{};
  float weight{};
  std::uint32_t flags{};
};

struct esdf_voxel {
  float distance{};
  std::uint32_t flags{};
};

struct tsdf_tile {
  std::size_t volume{};
  std::int32_t x{};
  std::int32_t y{};
  std::int32_t z{};
  std::array<tsdf_voxel, tile_voxels> voxels{};
};

struct esdf_tile {
  std::size_t volume{};
  std::int32_t x{};
  std::int32_t y{};
  std::int32_t z{};
  std::array<esdf_voxel, tile_voxels> voxels{};
};

struct field_volume {
  std::size_t pose_node{};
  double voxel_size{};
  bool has_tsdf{};
  bool has_esdf{};
};

struct registration_sample {
  std::size_t group{};
  vector3 reference_local_point{};
  double reference_distance{};
  double reference_tsdf_weight{1.0};
  std::uint64_t ordinal{};
};

struct registration_group {
  std::size_t node_i{};
  std::size_t node_j{};
  std::size_t volume_i{};
  std::size_t volume_j{};
  field_kind field{field_kind::esdf};
  std::size_t sample_begin{};
  std::size_t sample_count{};
  double direction_weight{1.0};
  std::uint64_t diagnostic_id{};
};

struct registration_model {
  registration_profile profile{registration_profile::uniform_trilinear};
  loss robust_loss{};
  double minimum_tsdf_weight{1.0e-6};
  double maximum_confidence_weight{10000.0};
  double minimum_gradient_norm{1.0e-6};
  double maximum_absolute_residual_m{};
  bool allow_nearest_fallback{};
};

struct problem {
  std::vector<node> nodes;
  std::vector<relative_factor> relative_factors;
  std::vector<absolute_factor> absolute_factors;
  std::vector<registration_group> registration_groups;
  std::vector<registration_sample> registration_samples;
  std::vector<field_volume> volumes;
  std::vector<tsdf_tile> tsdf_tiles;
  std::vector<esdf_tile> esdf_tiles;
  registration_model registration{};
};

struct field_snapshot;

namespace detail {
struct prepared_storage;
struct prepared_problem_access;
}

class prepared_problem {
public:
  prepared_problem() = default;

  bool empty() const noexcept;
  std::size_t node_count() const noexcept;
  std::size_t active_node_count() const noexcept;
  bool has_unanchored_components() const noexcept;
  const std::string& hash() const noexcept;

private:
  explicit prepared_problem(std::shared_ptr<const detail::prepared_storage> storage);

  std::shared_ptr<const detail::prepared_storage> storage_;

  friend struct detail::prepared_problem_access;
};

operation_result<prepared_problem> prepare(problem value);

}
