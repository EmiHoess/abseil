#pragma once

#include <seil/problem.hpp>

#include <cstddef>
#include <cstdint>

namespace seil {

enum class field_interpolation : std::uint32_t {
  nearest,
  trilinear
};

struct field_sample {
  double distance{};
  double weight{};
};

struct field_query {
  field_sample sample{};
  vector3 gradient{};
  bool gradient_valid{};
};

operation_result<field_query> query_field(
  const prepared_problem& value,
  field_kind field,
  std::size_t volume,
  vector3 point,
  field_interpolation interpolation,
  bool calculate_gradient = true
);

}
