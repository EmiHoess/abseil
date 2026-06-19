#include "../src/linear_solver.hpp"

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

seil::problem chain(bool anchored)
{
  seil::problem graph;
  graph.nodes.push_back({1, {}, anchored});
  graph.nodes.push_back({2, {2.0, 0.0, 0.0, 0.0}, false});
  graph.relative_factors.push_back({0, 1, {1.0, 0.0, 0.0, 0.0}});
  return graph;
}

}

int main()
{
  auto graph = chain(true);
  auto prepared = seil::prepare(graph);
  require(prepared.has_value(), "linear graph preparation");
  const auto* storage = seil::detail::prepared_problem_access::get(*prepared);
  const std::array poses{graph.nodes[0].initial, graph.nodes[1].initial};
  auto system = seil::detail::build_sparse_system(*storage, poses);
  require(system.has_value(), "linear system construction");
  auto solution = seil::detail::solve_sparse_system(
    storage->topology, *system, 1.0e-3, {});
  require(solution.converged && solution.update.size() == 4,
    "pcg converges");
  require(solution.update[0] < 0.0, "pcg correction direction");

  graph = chain(false);
  prepared = seil::prepare(graph);
  require(prepared.has_value(), "unanchored graph preparation");
  storage = seil::detail::prepared_problem_access::get(*prepared);
  const std::array free_poses{graph.nodes[0].initial, graph.nodes[1].initial};
  system = seil::detail::build_sparse_system(*storage, free_poses);
  require(system.has_value(), "unanchored system construction");
  solution = seil::detail::solve_sparse_system(storage->topology, *system,
    1.0e-3, {});
  require(!solution.converged && solution.stop == seil::termination::singular_graph,
    "unanchored graph rejected");
}
