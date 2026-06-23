#include <seil/seil.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

void require(bool condition, const char* message)
{
  if(!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

seil::prepared_problem graph()
{
  seil::problem value;
  value.nodes.push_back({1, {}, true});
  value.nodes.push_back({2, {3.0, 1.0, 0.0, 0.2}, false});
  value.relative_factors.push_back({0, 1, {1.0, 0.0, 0.0, 0.0}});
  auto prepared = seil::prepare(std::move(value));
  require(prepared.has_value(), "parity graph preparation");
  return std::move(*prepared);
}

}

int main()
{
  const auto prepared = graph();
  auto native = seil::solve(prepared, {});
  require(native.has_value() && native->solution_usable,
    "native parity solve");
  if(!seil::backend_available(seil::backend::ceres_cpu))
    return 0;
  seil::solve_options options;
  options.selected_backend = seil::backend::ceres_cpu;
  options.specific = seil::ceres_cpu_options{};
  auto ceres = seil::solve(prepared, options);
  require(ceres.has_value() && ceres->solution_usable, "ceres parity solve");
  require(std::abs(ceres->poses[1].x - native->poses[1].x) < 1.0e-5 &&
    std::abs(ceres->poses[1].y - native->poses[1].y) < 1.0e-5 &&
    std::abs(ceres->final_cost.total() - native->final_cost.total()) < 1.0e-8,
    "ceres native parity");
}
