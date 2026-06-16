#include <seil/seil.hpp>

#include <type_traits>

int main()
{
  using namespace seil;

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

  return 0;
}
