#include <abseil/abseil.hpp>

#include <cstdlib>

int main()
{
  abseil::problem value;
  value.nodes.push_back({1, {}, true});
  auto prepared = abseil::prepare(std::move(value));
  if(!prepared)
    return EXIT_FAILURE;
  auto solved = abseil::solve(*prepared, {});
  if(!solved || !solved->solution_usable || solved->poses.size() != 1 ||
    solved->stop != abseil::termination::no_free_nodes)
    return EXIT_FAILURE;
  return EXIT_SUCCESS;
}
