#include <seil/solver.hpp>

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

seil::prepared_problem graph()
{
  seil::problem value;
  value.nodes.push_back({1, {}, true});
  auto prepared = seil::prepare(std::move(value));
  require(prepared.has_value(), "dispatch graph preparation");
  return std::move(*prepared);
}

}

int main()
{
  require(seil::parse_backend("native_cpu") == seil::backend::native_cpu,
    "native backend lookup");
  require(seil::parse_backend("ceres_cpu") == seil::backend::ceres_cpu,
    "ceres backend lookup");
  require(!seil::parse_backend("Native_CPU"), "backend lookup is exact");
  require(seil::backend_name(seil::backend::native_cuda_cold) ==
    "native_cuda_cold", "backend name");
  require(seil::termination_name(seil::termination::converged_cost) ==
    "converged_cost", "termination name");
  const auto available = seil::available_backends();
  require(!available.empty() && available.front() == seil::backend::native_cpu,
    "native backend enumeration");
  require((std::find(available.begin(), available.end(), seil::backend::ceres_cpu) !=
    available.end()) == seil::backend_available(seil::backend::ceres_cpu),
    "optional backend enumeration");

  const auto prepared = graph();
  auto solved = seil::solve(prepared, {});
  require(solved.has_value() && solved->stop == seil::termination::no_free_nodes,
    "native dispatch");
  require(solved->poses.size() == prepared.node_count(),
    "native result pose count");
  seil::solve_options mismatch;
  mismatch.specific = seil::ceres_cpu_options{};
  require(!seil::solve(prepared, mismatch), "backend option mismatch");
  seil::solve_options unavailable;
  unavailable.selected_backend = seil::backend::ceres_cpu;
  unavailable.specific = seil::ceres_cpu_options{};
  auto result = seil::solve(prepared, unavailable);
  if(seil::backend_available(seil::backend::ceres_cpu))
    require(result.has_value(), "available optional backend dispatch");
  else
    require(!result && result.error().code == seil::operation_code::backend_unavailable,
      "unavailable backend error");
}
