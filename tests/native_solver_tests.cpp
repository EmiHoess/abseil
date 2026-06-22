#include "../src/internal.hpp"

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

seil::prepared_problem chain(bool free_second = true)
{
  seil::problem graph;
  graph.nodes.push_back({1, {}, true});
  graph.nodes.push_back({2, {3.0, 1.0, 0.0, 0.2}, !free_second});
  graph.relative_factors.push_back({0, 1, {1.0, 0.0, 0.0, 0.0}});
  auto prepared = seil::prepare(std::move(graph));
  require(prepared.has_value(), "native graph preparation");
  return std::move(*prepared);
}

}

int main()
{
  auto problem = chain();
  auto solved = seil::detail::solve_native_cpu(problem, {}, {});
  require(solved.has_value() && solved->solution_usable,
    "native solve succeeds");
  require(solved->final_cost.total() < solved->initial_cost.total(),
    "native objective decreases");
  require(std::abs(solved->poses[1].x - 1.0) < 1.0e-4 &&
    std::abs(solved->poses[1].y) < 1.0e-4,
    "native pose optimized");

  problem = chain(false);
  solved = seil::detail::solve_native_cpu(problem, {}, {});
  require(solved.has_value() && solved->solution_usable &&
    solved->stop == seil::termination::no_free_nodes,
    "fixed problem terminates");

  seil::native_cpu_options invalid;
  invalid.lm.initial_damping = 0.0;
  require(!seil::detail::solve_native_cpu(problem, {}, invalid),
    "invalid native options rejected");

  seil::detail::native_termination_state state;
  state.gradient_norm = 1.0;
  state.elapsed_seconds = 31.0;
  require(seil::detail::native_termination(state, {}, {}).value() ==
    seil::termination::maximum_time, "time termination policy");
  state.elapsed_seconds = 0.0;
  state.gradient_norm = 0.0;
  require(seil::detail::native_termination(state, {}, {}).value() ==
    seil::termination::converged_gradient, "gradient termination policy");
  state.gradient_norm = 1.0;
  state.step_complete = true;
  state.accepted = true;
  state.update_norm = 0.0;
  require(seil::detail::native_termination(state, {}, {}).value() ==
    seil::termination::converged_update, "update termination policy");
  state.accepted = false;
  state.rejected_steps = 8;
  require(seil::detail::native_termination(state, {}, {}).value() ==
    seil::termination::rejection_limit, "rejection termination policy");
}
