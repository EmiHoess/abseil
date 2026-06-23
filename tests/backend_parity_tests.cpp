#include <seil/seil.hpp>

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

seil::prepared_problem graph()
{
  seil::problem value;
  value.nodes.push_back({1, {}, true});
  value.nodes.push_back({2, {3.0, 1.0, 0.0, 0.2}, false});
  value.relative_factors.push_back({0, 1, {1.0, 0.0, 0.0, 0.0}});
  value.absolute_factors.push_back({0, 1, {1.0, 0.0, 0.0, 0.0}});
  value.volumes.push_back({0, 1.0, true, true});
  value.volumes.push_back({1, 1.0, true, true});
  seil::tsdf_tile tsdf;
  seil::esdf_tile esdf;
  tsdf.volume = 1;
  esdf.volume = 1;
  for(std::size_t z = 0; z < seil::tile_edge; ++z)
    for(std::size_t y = 0; y < seil::tile_edge; ++y)
      for(std::size_t x = 0; x < seil::tile_edge; ++x) {
        const std::size_t index = x + 8U * (y + 8U * z);
        const float distance = static_cast<float>(x);
        tsdf.voxels[index] = {distance, 2.0F, 1U};
        esdf.voxels[index] = {distance, 1U};
      }
  value.tsdf_tiles.push_back(tsdf);
  value.esdf_tiles.push_back(esdf);
  value.registration_samples.push_back({0, {3.5, 2.5, 2.5}, 2.5, 2.0, 0});
  value.registration_groups.push_back({0, 1, 0, 1, seil::field_kind::esdf,
    0, 1, 0.25, 0});
  auto prepared = seil::prepare(std::move(value));
  require(prepared.has_value(), "parity graph preparation");
  return std::move(*prepared);
}

}

int main()
{
  const auto prepared = graph();
  auto native = seil::solve(prepared, {});
  require(native.has_value() && native->solution_usable,
    "native parity solve");
  if(!seil::backend_available(seil::backend::ceres_cpu))
    return 0;
  seil::solve_options options;
  options.selected_backend = seil::backend::ceres_cpu;
  options.specific = seil::ceres_cpu_options{};
  auto ceres = seil::solve(prepared, options);
  require(ceres.has_value() && ceres->solution_usable, "ceres parity solve");
  require(std::abs(ceres->poses[1].x - native->poses[1].x) < 1.0e-4 &&
    std::abs(ceres->poses[1].y - native->poses[1].y) < 1.0e-4 &&
    std::abs(ceres->final_cost.total() - native->final_cost.total()) < 1.0e-7 &&
    ceres->registration.accepted_rows == native->registration.accepted_rows,
    "ceres native parity");
}
