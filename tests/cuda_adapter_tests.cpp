#include "../src/cuda_backend.hpp"

#include <seil/problem.hpp>
#include <seil/solver.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

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
  require(seil_cuda_abi_version() == seil_cuda_abi_version_value,
    "CUDA ABI version");
  int32_t count = -1;
  const int status = seil_cuda_device_count(&count);
  require(status == 0 || count == 0, "CUDA device query status");
  seil::problem graph;
  graph.nodes.push_back({1, {}, true});
  graph.nodes.push_back({2, {2.0, 0.0, 0.0, 0.0}, false});
  graph.relative_factors.push_back({0, 1, {1.0, 0.0, 0.0, 0.0}});
  auto prepared = seil::prepare(std::move(graph));
  require(prepared.has_value(), "CUDA adapter graph preparation");
#if defined(SEIL_HAS_CUDA)
  require(seil::detail::cuda_packing::index(42) == 42,
    "CUDA checked index");
  require(!seil::detail::cuda_packing::index(
    static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) + 1U),
    "CUDA index overflow");
  require(!seil::detail::cuda_packing::add_bytes(
    std::numeric_limits<std::uint64_t>::max(), 1, 1),
    "CUDA byte overflow");
  auto packed = seil::detail::pack_cuda_problem(*prepared);
  require(packed.has_value() && packed->nodes.size() == 2 &&
    packed->relative_edges.size() == 1 && packed->byte_count > 0,
    "CUDA snapshot packing");
  if(status == 0 && count > 0) {
    seil::native_cuda_cold_options public_options;
    public_options.device_index = 0;
    auto public_result = seil::solve_cuda_checked(*prepared, {}, public_options);
    require(public_result.has_value() && public_result->solution_usable &&
      public_result->selected_backend == seil::backend::native_cuda_cold,
      "checked CUDA solve");
    require(std::abs(public_result->poses[1].x - 1.0) < 1.0e-10,
      "checked CUDA pose");
    seil::solve_options dispatched_options;
    dispatched_options.selected_backend = seil::backend::native_cuda_cold;
    dispatched_options.specific = public_options;
    auto dispatched = seil::solve(*prepared, dispatched_options);
    require(dispatched.has_value() && dispatched->solution_usable,
      "generic CUDA dispatch");
    seil::native_cuda_cold_options missing_device;
    missing_device.device_index = count;
    auto unavailable = seil::solve_cuda_checked(*prepared, {}, missing_device);
    require(!unavailable && unavailable.error().stage ==
      seil::operation_stage::device_query, "checked CUDA device error");
    seil::native_cuda_cold_options limited;
    limited.device_index = 0;
    limited.memory_budget_bytes = 1;
    auto exhausted = seil::solve_cuda_checked(*prepared, {}, limited);
    require(!exhausted && exhausted.error().stage == seil::operation_stage::upload &&
      exhausted.error().code == seil::operation_code::resource_exhausted,
      "checked CUDA budget error");
    auto recovered = seil::solve_cuda_checked(*prepared, {}, public_options);
    require(recovered.has_value() && recovered->solution_usable,
      "checked CUDA recovery");

    seil_cuda_options options{};
    options.abi_version = seil_cuda_abi_version_value;
    options.struct_size = sizeof(options);
    options.device_index = 0;
    options.maximum_iterations = 20;
    options.maximum_translation_step_m = 2.0;
    options.maximum_yaw_step_rad = 0.5;
    options.minimum_update_norm = 1.0e-8;
    options.registration_profile = 1;
    options.registration_loss = {seil_cuda_loss_none, 1.0};
    seil_cuda_context* context = nullptr;
    require(seil_cuda_create(&options, &context) == 0 && context != nullptr,
      "CUDA context creation");
    require(seil_cuda_upload_problem(context, &packed->view) == 0,
      "CUDA snapshot upload");
    seil_cuda_summary summary{};
    require(seil_cuda_solve_stage(context, &summary) == 0,
      "CUDA solve stage status");
    require(std::abs(summary.initial_relative_cost - 0.5) < 1.0e-10,
      "CUDA relative evaluation");
    require(summary.solution_usable != 0 &&
      summary.final_relative_cost < summary.initial_relative_cost,
      "CUDA sparse solve");
    require(summary.synchronization_count == 1,
      "CUDA compact synchronization");
    size_t pose_count = 0;
    require(seil_cuda_download_poses(context, nullptr, 0, &pose_count) == 0 &&
      pose_count == 2, "CUDA pose count");
    seil_cuda_pose4 poses[2]{};
    require(seil_cuda_download_poses(context, poses, 2, &pose_count) == 0 &&
      std::abs(poses[1].x - 1.0) < 1.0e-10, "CUDA pose download");
    size_t iteration_count = 0;
    require(seil_cuda_download_iterations(context, nullptr, 0,
      &iteration_count) == 0 && iteration_count == 1,
      "CUDA iteration count");
    require(seil_cuda_destroy(context) == 0, "CUDA context cleanup");
  }
#endif
}
