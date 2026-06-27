#include "../src/cuda_backend.hpp"

#include <abseil/problem.hpp>
#include <abseil/solver.hpp>

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
  require(abseil_cuda_abi_version() == abseil_cuda_abi_version_value,
    "CUDA ABI version");
  int32_t count = -1;
  const int status = abseil_cuda_device_count(&count);
  require(status == 0 || count == 0, "CUDA device query status");
  abseil::problem graph;
  graph.nodes.push_back({1, {}, true});
  graph.nodes.push_back({2, {2.0, 0.0, 0.0, 0.0}, false});
  graph.relative_factors.push_back({0, 1, {1.0, 0.0, 0.0, 0.0}});
  auto prepared = abseil::prepare(std::move(graph));
  require(prepared.has_value(), "CUDA adapter graph preparation");
#if defined(ABSEIL_HAS_CUDA)
  require(abseil::detail::cuda_packing::index(42) == 42,
    "CUDA checked index");
  require(!abseil::detail::cuda_packing::index(
    static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) + 1U),
    "CUDA index overflow");
  require(!abseil::detail::cuda_packing::add_bytes(
    std::numeric_limits<std::uint64_t>::max(), 1, 1),
    "CUDA byte overflow");
  auto packed = abseil::detail::pack_cuda_problem(*prepared);
  require(packed.has_value() && packed->nodes.size() == 2 &&
    packed->relative_edges.size() == 1 && packed->byte_count > 0,
    "CUDA snapshot packing");
  if(status == 0 && count > 0) {
    abseil::native_cuda_cold_options public_options;
    public_options.device_index = 0;
    auto public_result = abseil::solve_cuda_checked(*prepared, {}, public_options);
    require(public_result.has_value() && public_result->solution_usable &&
      public_result->selected_backend == abseil::backend::native_cuda_cold,
      "checked CUDA solve");
    require(std::abs(public_result->poses[1].x - 1.0) < 1.0e-10,
      "checked CUDA pose");
    abseil::solve_options dispatched_options;
    dispatched_options.selected_backend = abseil::backend::native_cuda_cold;
    dispatched_options.specific = public_options;
    auto dispatched = abseil::solve(*prepared, dispatched_options);
    require(dispatched.has_value() && dispatched->solution_usable,
      "generic CUDA dispatch");
    abseil::native_cuda_cold_options missing_device;
    missing_device.device_index = count;
    auto unavailable = abseil::solve_cuda_checked(*prepared, {}, missing_device);
    require(!unavailable && unavailable.error().stage ==
      abseil::operation_stage::device_query, "checked CUDA device error");
    abseil::native_cuda_cold_options limited;
    limited.device_index = 0;
    limited.memory_budget_bytes = 1;
    auto exhausted = abseil::solve_cuda_checked(*prepared, {}, limited);
    require(!exhausted && exhausted.error().stage == abseil::operation_stage::upload &&
      exhausted.error().code == abseil::operation_code::resource_exhausted,
      "checked CUDA budget error");
    auto recovered = abseil::solve_cuda_checked(*prepared, {}, public_options);
    require(recovered.has_value() && recovered->solution_usable,
      "checked CUDA recovery");

    abseil_cuda_options options{};
    options.abi_version = abseil_cuda_abi_version_value;
    options.struct_size = sizeof(options);
    options.device_index = 0;
    options.maximum_iterations = 20;
    options.maximum_translation_step_m = 2.0;
    options.maximum_yaw_step_rad = 0.5;
    options.minimum_update_norm = 1.0e-8;
    options.registration_profile = 1;
    options.registration_loss = {abseil_cuda_loss_none, 1.0};
    abseil_cuda_context* context = nullptr;
    require(abseil_cuda_create(&options, &context) == 0 && context != nullptr,
      "CUDA context creation");
    require(abseil_cuda_upload_problem(context, &packed->view) == 0,
      "CUDA snapshot upload");
    abseil_cuda_summary summary{};
    require(abseil_cuda_solve_stage(context, &summary) == 0,
      "CUDA solve stage status");
    require(std::abs(summary.initial_relative_cost - 0.5) < 1.0e-10,
      "CUDA relative evaluation");
    require(summary.solution_usable != 0 &&
      summary.final_relative_cost < summary.initial_relative_cost,
      "CUDA sparse solve");
    require(summary.synchronization_count == 1,
      "CUDA compact synchronization");
    size_t pose_count = 0;
    require(abseil_cuda_download_poses(context, nullptr, 0, &pose_count) == 0 &&
      pose_count == 2, "CUDA pose count");
    abseil_cuda_pose4 poses[2]{};
    require(abseil_cuda_download_poses(context, poses, 2, &pose_count) == 0 &&
      std::abs(poses[1].x - 1.0) < 1.0e-10, "CUDA pose download");
    size_t iteration_count = 0;
    require(abseil_cuda_download_iterations(context, nullptr, 0,
      &iteration_count) == 0 && iteration_count == 1,
      "CUDA iteration count");
    require(abseil_cuda_destroy(context) == 0, "CUDA context cleanup");

    options.reduction_policy = abseil_cuda_reduction_atomic_fast;
    context = nullptr;
    require(abseil_cuda_create(&options, &context) == 0 && context != nullptr,
      "atomic CUDA context creation");
    require(abseil_cuda_upload_problem(context, &packed->view) == 0,
      "atomic CUDA snapshot upload");
    abseil_cuda_summary atomic_summary{};
    require(abseil_cuda_solve_stage(context, &atomic_summary) == 0 &&
      std::abs(atomic_summary.initial_relative_cost -
        summary.initial_relative_cost) < 1.0e-10 &&
      std::abs(atomic_summary.final_relative_cost -
        summary.final_relative_cost) < 1.0e-10,
      "atomic CUDA reduction parity");
    require(abseil_cuda_destroy(context) == 0, "atomic CUDA context cleanup");

    options.abi_version = 0;
    require(abseil_cuda_create(&options, &context) != 0 && context == nullptr,
      "CUDA ABI mismatch rejection");
  }
#endif
}
