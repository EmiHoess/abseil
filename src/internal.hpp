#pragma once

#include <seil/problem.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
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
  std::string hash;
};

struct prepared_problem_access {
  static prepared_problem make(std::shared_ptr<const prepared_storage> storage);
  static const prepared_storage* get(const prepared_problem& value) noexcept;
};

}
