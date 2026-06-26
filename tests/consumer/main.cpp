#include <seil/seil.hpp>

#include <cstdlib>

int main()
{
  seil::problem value;
  value.nodes.push_back({1, {}, true});
  auto prepared = seil::prepare(std::move(value));
  if(!prepared)
    return EXIT_FAILURE;
  auto solved = seil::solve(*prepared, {});
  if(!solved || !solved->solution_usable || solved->poses.size() != 1 ||
    solved->stop != seil::termination::no_free_nodes)
    return EXIT_FAILURE;
  return EXIT_SUCCESS;
}
