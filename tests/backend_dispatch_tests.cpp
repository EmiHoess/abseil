#include <abseil/solver.hpp>

#include <algorithm>
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

abseil::prepared_problem graph()
{
  abseil::problem value;
  value.nodes.push_back({1, {}, true});
  auto prepared = abseil::prepare(std::move(value));
  require(prepared.has_value(), "dispatch graph preparation");
  return std::move(*prepared);
}

}

int main()
{
  require(abseil::parse_backend("native_cpu") == abseil::backend::native_cpu,
    "native backend lookup");
  require(abseil::parse_backend("ceres_cpu") == abseil::backend::ceres_cpu,
    "ceres backend lookup");
  require(!abseil::parse_backend("Native_CPU"), "backend lookup is exact");
  require(abseil::backend_name(abseil::backend::native_cuda_cold) ==
    "native_cuda_cold", "backend name");
  require(abseil::termination_name(abseil::termination::converged_cost) ==
    "converged_cost", "termination name");
  const auto available = abseil::available_backends();
  require(!available.empty() && available.front() == abseil::backend::native_cpu,
    "native backend enumeration");
  require((std::find(available.begin(), available.end(), abseil::backend::ceres_cpu) !=
    available.end()) == abseil::backend_available(abseil::backend::ceres_cpu),
    "optional backend enumeration");

  const auto prepared = graph();
  auto solved = abseil::solve(prepared, {});
  require(solved.has_value() && solved->stop == abseil::termination::no_free_nodes,
    "native dispatch");
  require(solved->poses.size() == prepared.node_count(),
    "native result pose count");
  abseil::solve_options mismatch;
  mismatch.specific = abseil::ceres_cpu_options{};
  require(!abseil::solve(prepared, mismatch), "backend option mismatch");
  abseil::solve_options unavailable;
  unavailable.selected_backend = abseil::backend::ceres_cpu;
  unavailable.specific = abseil::ceres_cpu_options{};
  auto result = abseil::solve(prepared, unavailable);
  if(abseil::backend_available(abseil::backend::ceres_cpu))
    require(result.has_value(), "available optional backend dispatch");
  else
    require(!result && result.error().code == abseil::operation_code::backend_unavailable,
      "unavailable backend error");
}
