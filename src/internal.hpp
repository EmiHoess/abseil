#pragma once

#include <seil/problem.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace seil::detail {

struct prepared_storage {
  problem value;
  std::vector<std::int32_t> active_index;
  std::size_t active_count{};
  std::size_t unanchored_components{};
  std::string hash;
};

struct prepared_problem_access {
  static prepared_problem make(std::shared_ptr<const prepared_storage> storage);
  static const prepared_storage* get(const prepared_problem& value) noexcept;
};

}
