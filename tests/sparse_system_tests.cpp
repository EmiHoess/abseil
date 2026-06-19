#include "../src/sparse_system.hpp"

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

}

int main()
{
  seil::problem graph;
  graph.nodes.push_back({1, {}, true});
  graph.nodes.push_back({2, {2.0, 0.0, 0.0, 0.0}, false});
  graph.nodes.push_back({3, {4.0, 0.0, 0.0, 0.0}, false});
  graph.relative_factors.push_back({0, 1, {1.0, 0.0, 0.0, 0.0}});
  graph.relative_factors.push_back({1, 2, {1.0, 0.0, 0.0, 0.0}});
  auto prepared = seil::prepare(graph);
  require(prepared.has_value(), "sparse graph preparation");
  const auto* storage = seil::detail::prepared_problem_access::get(*prepared);
  const std::array poses{
    graph.nodes[0].initial,
    graph.nodes[1].initial,
    graph.nodes[2].initial
  };
  auto system = seil::detail::build_sparse_system(*storage, poses);
  require(system.has_value(), "sparse system construction");
  require(system->gradient.size() == 8, "sparse gradient dimension");
  require(system->blocks.size() == 3 && system->hessian.size() == 48,
    "sparse block dimensions");
  require(system->blocks[0].row == 0 && system->blocks[0].column == 0 &&
    system->blocks[1].row == 0 && system->blocks[1].column == 1 &&
    system->blocks[2].row == 1 && system->blocks[2].column == 1,
    "deterministic block order");
  require(system->cost.relative > 0.0, "sparse system cost");
  for(double value : system->hessian)
    require(std::isfinite(value), "finite sparse hessian");
}
