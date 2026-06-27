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

abseil::prepared_problem chain(bool free_second = true)
{
  abseil::problem graph;
  graph.nodes.push_back({1, {}, true});
  graph.nodes.push_back({2, {3.0, 1.0, 0.0, 0.2}, !free_second});
  graph.relative_factors.push_back({0, 1, {1.0, 0.0, 0.0, 0.0}});
  auto prepared = abseil::prepare(std::move(graph));
  require(prepared.has_value(), "native graph preparation");
  return std::move(*prepared);
}

}

int main()
{
  auto problem = chain();
  auto solved = abseil::detail::solve_native_cpu(problem, {}, {});
  require(solved.has_value() && solved->solution_usable,
    "native solve succeeds");
  require(solved->final_cost.total() < solved->initial_cost.total(),
    "native objective decreases");
  require(std::abs(solved->poses[1].x - 1.0) < 1.0e-4 &&
    std::abs(solved->poses[1].y) < 1.0e-4,
    "native pose optimized");
  require(!solved->iterations.empty() && solved->telemetry.accepted_steps > 0 &&
    solved->telemetry.accepted_steps + solved->telemetry.rejected_steps ==
      solved->iterations.size(), "native iteration telemetry");
  require(solved->telemetry.total_ms > 0.0 &&
    solved->telemetry.implementation == "abseil_native_cpu" &&
    solved->telemetry.device == "host", "native stage telemetry");
  require(solved->iterations.front().linear_solver_seconds >= 0.0 &&
    solved->iterations.front().iteration_seconds >= 0.0,
    "native timing telemetry");
  const auto repeated = abseil::detail::solve_native_cpu(problem, {}, {});
  require(repeated.has_value() && repeated->poses.size() == solved->poses.size() &&
    std::abs(repeated->final_cost.total() - solved->final_cost.total()) < 1.0e-12 &&
    std::abs(repeated->poses[1].x - solved->poses[1].x) < 1.0e-12,
    "native solve is deterministic");

  abseil::common_options bounded;
  bounded.maximum_iterations = 1;
  abseil::native_cpu_options bounded_options;
  bounded_options.lm.minimum_cost_change = 0.0;
  bounded_options.lm.minimum_gradient_norm = 0.0;
  bounded_options.lm.minimum_update_norm = 0.0;
  auto bounded_result = abseil::detail::solve_native_cpu(
    problem, bounded, bounded_options);
  require(bounded_result.has_value() &&
    bounded_result->stop == abseil::termination::maximum_iterations,
    "maximum iteration termination");

  problem = chain(false);
  solved = abseil::detail::solve_native_cpu(problem, {}, {});
  require(solved.has_value() && solved->solution_usable &&
    solved->stop == abseil::termination::no_free_nodes,
    "fixed problem terminates");

  abseil::native_cpu_options invalid;
  invalid.lm.initial_damping = 0.0;
  require(!abseil::detail::solve_native_cpu(problem, {}, invalid),
    "invalid native options rejected");

  abseil::detail::native_termination_state state;
  state.gradient_norm = 1.0;
  state.elapsed_seconds = 31.0;
  require(abseil::detail::native_termination(state, {}, {}).value() ==
    abseil::termination::maximum_time, "time termination policy");
  state.elapsed_seconds = 0.0;
  state.gradient_norm = 0.0;
  require(abseil::detail::native_termination(state, {}, {}).value() ==
    abseil::termination::converged_gradient, "gradient termination policy");
  state.gradient_norm = 1.0;
  state.step_complete = true;
  state.accepted = true;
  state.update_norm = 0.0;
  require(abseil::detail::native_termination(state, {}, {}).value() ==
    abseil::termination::converged_update, "update termination policy");
  state.accepted = false;
  state.rejected_steps = 8;
  require(abseil::detail::native_termination(state, {}, {}).value() ==
    abseil::termination::rejection_limit, "rejection termination policy");
  state.accepted = true;
  state.rejected_steps = 0;
  state.update_norm = 1.0;
  state.cost_change = 0.0;
  require(abseil::detail::native_termination(state, {}, {}).value() ==
    abseil::termination::converged_cost, "cost termination policy");
}
