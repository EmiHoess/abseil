#include <abseil/abseil.hpp>

#include <type_traits>

namespace {

abseil::problem anchored_problem(double x)
{
  abseil::problem value;
  value.nodes.push_back({1, {0.0, 0.0, 0.0, 0.0}, true});
  value.nodes.push_back({2, {x, 0.0, 0.0, 0.0}, false});
  value.relative_factors.push_back({0, 1, {1.0, 0.0, 0.0, 0.0}});
  return value;
}

}

int main()
{
  using namespace abseil;

  static_assert(std::is_same_v<decltype(prepare(problem{})), operation_result<prepared_problem>>);
  static_assert(std::variant_size_v<backend_options> == 3);
  static_assert(tile_voxels == 512);

  problem value;
  value.nodes.push_back(node{7, pose4{1.0, 2.0, 3.0, 0.25}, true});
  value.registration.profile = registration_profile::confidence_weighted;

  solve_options options;
  if(options.selected_backend != backend::native_cpu)
    return 1;
  if(!std::holds_alternative<native_cpu_options>(options.specific))
    return 2;

  cost_breakdown cost{1.0, 2.0, 3.0};
  if(cost.total() != 6.0)
    return 3;
  if(value.nodes.front().id != 7)
    return 4;

  operation_result<prepared_problem> prepared = prepare(std::move(value));
  if(!prepared)
    return 5;

  problem empty;
  operation_result<prepared_problem> empty_result = prepare(std::move(empty));
  if(empty_result || empty_result.error().stage != operation_stage::preparation)
    return 6;

  problem duplicate;
  duplicate.nodes.push_back(node{9, {}, true});
  duplicate.nodes.push_back(node{9, {}, false});
  if(prepare(std::move(duplicate)))
    return 7;

  operation_result<prepared_problem> first = prepare(anchored_problem(1.0));
  operation_result<prepared_problem> repeated = prepare(anchored_problem(1.0));
  operation_result<prepared_problem> changed = prepare(anchored_problem(2.0));
  if(!first || !repeated || !changed)
    return 8;
  if(first->empty() || first->node_count() != 2 || first->active_node_count() != 1)
    return 9;
  if(first->has_unanchored_components())
    return 10;
  if(first->hash().empty() || first->hash() != repeated->hash())
    return 11;
  if(first->hash() == changed->hash())
    return 12;

  problem unanchored;
  unanchored.nodes.push_back({3, {}, false});
  operation_result<prepared_problem> disconnected = prepare(std::move(unanchored));
  if(!disconnected || !disconnected->has_unanchored_components())
    return 13;

  problem malformed = anchored_problem(1.0);
  malformed.relative_factors[0].node_j = 4;
  if(prepare(std::move(malformed)))
    return 14;
  malformed = anchored_problem(1.0);
  malformed.volumes.push_back({3, 0.0, true, false});
  if(prepare(std::move(malformed)))
    return 15;
  malformed = anchored_problem(1.0);
  malformed.volumes.push_back({0, 1.0, true, false});
  malformed.tsdf_tiles.push_back({0, 1, 2, 3, {}});
  malformed.tsdf_tiles.push_back({0, 1, 2, 3, {}});
  if(prepare(std::move(malformed)))
    return 16;

  return 0;
}
