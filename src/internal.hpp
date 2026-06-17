#pragma once

#include <seil/problem.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace seil::detail {

struct graph_component {
  bool has_active{};
  bool has_constant{};
};

struct sparse_candidate {
  std::int32_t row{};
  std::int32_t column{};
};

struct tile_key {
  std::size_t volume{};
  std::int32_t x{};
  std::int32_t y{};
  std::int32_t z{};

  friend bool operator==(const tile_key&, const tile_key&) = default;
};

struct tile_key_hash {
  std::size_t operator()(const tile_key& key) const noexcept
  {
    std::size_t value = key.volume + 0x9e3779b9U;
    const auto combine = [&value](std::uint32_t item) {
      value ^= static_cast<std::size_t>(item) + 0x9e3779b9U +
        (value << 6U) + (value >> 2U);
    };
    combine(static_cast<std::uint32_t>(key.x));
    combine(static_cast<std::uint32_t>(key.y));
    combine(static_cast<std::uint32_t>(key.z));
    return value;
  }
};

struct topology_index {
  std::vector<std::int32_t> active_node;
  std::vector<std::size_t> component_of_node;
  std::vector<graph_component> components;
  std::vector<sparse_candidate> sparse_candidates;
  std::size_t active_count{};
  std::size_t unanchored_components{};
};

struct prepared_storage {
  problem value;
  topology_index topology;
  std::unordered_map<tile_key, std::size_t, tile_key_hash> tsdf_index;
  std::unordered_map<tile_key, std::size_t, tile_key_hash> esdf_index;
  std::string hash;
};

struct prepared_problem_access {
  static prepared_problem make(std::shared_ptr<const prepared_storage> storage);
  static const prepared_storage* get(const prepared_problem& value) noexcept;
};

}
