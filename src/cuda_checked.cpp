#include "cuda_backend.hpp"
#include "internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>

namespace seil {
namespace {

struct context_deleter {
  void operator()(seil_cuda_context* context) const noexcept
  {
    seil_cuda_destroy(context);
  }
};

using owned_context = std::unique_ptr<seil_cuda_context, context_deleter>;

operation_error cuda_error(operation_stage stage, int status, std::string fallback)
{
  const char* native_message = seil_cuda_last_error();
  std::string message = native_message != nullptr && native_message[0] != '\0' ?
    native_message : std::move(fallback);
  operation_code code = operation_code::cuda_failure;
  if(status == static_cast<int>(detail::cuda_stage_status::unavailable))
    code = operation_code::backend_unavailable;
  else if(status == static_cast<int>(detail::cuda_stage_status::invalid_abi) ||
    status == static_cast<int>(detail::cuda_stage_status::invalid_argument))
    code = operation_code::invalid_argument;
  else if(status == static_cast<int>(detail::cuda_stage_status::resource_exhausted))
    code = operation_code::resource_exhausted;
  return {code, stage, status, std::move(message)};
}

bool valid_options(const common_options& common, const native_cuda_cold_options& options)
{
  const auto& lm = options.lm;
  return common.maximum_iterations > 0 && common.maximum_seconds > 0.0 &&
    std::isfinite(common.maximum_seconds) && options.device_index >= 0 &&
    lm.maximum_rejected_steps > 0 && lm.pcg_maximum_iterations > 0 &&
    lm.initial_damping > 0.0 && lm.minimum_damping > 0.0 &&
    lm.maximum_damping >= lm.initial_damping &&
    lm.initial_damping >= lm.minimum_damping && lm.minimum_cost_change >= 0.0 &&
    lm.minimum_gradient_norm >= 0.0 && lm.minimum_update_norm >= 0.0 &&
    lm.maximum_translation_step_m > 0.0 && lm.maximum_yaw_step_rad > 0.0 &&
    lm.pcg_tolerance > 0.0;
}

seil_cuda_options pack_options(
  const detail::prepared_storage& storage,
  const common_options& common,
  const native_cuda_cold_options& options
)
{
  const auto& lm = options.lm;
  const auto& registration = storage.value.registration;
  return {
    seil_cuda_abi_version_value,
    sizeof(seil_cuda_options),
    options.device_index,
    options.memory_budget_bytes,
    static_cast<std::uint32_t>(std::min<std::size_t>(common.maximum_iterations,
      std::numeric_limits<std::uint32_t>::max())),
    static_cast<std::uint32_t>(std::min<std::size_t>(lm.maximum_rejected_steps,
      std::numeric_limits<std::uint32_t>::max())),
    static_cast<std::uint32_t>(std::min<std::size_t>(lm.pcg_maximum_iterations,
      std::numeric_limits<std::uint32_t>::max())),
    common.maximum_seconds,
    lm.initial_damping,
    lm.minimum_damping,
    lm.maximum_damping,
    lm.minimum_cost_change,
    lm.minimum_gradient_norm,
    lm.minimum_update_norm,
    lm.maximum_translation_step_m,
    lm.maximum_yaw_step_rad,
    lm.pcg_tolerance,
    registration.minimum_tsdf_weight,
    registration.maximum_confidence_weight,
    registration.minimum_gradient_norm,
    registration.maximum_absolute_residual_m,
    registration.allow_nearest_fallback ? 1U : 0U,
    seil_cuda_reduction_deterministic,
    static_cast<std::uint32_t>(registration.profile),
    {static_cast<std::uint32_t>(registration.robust_loss.kind),
      registration.robust_loss.scale}
  };
}

termination map_termination(std::uint32_t value)
{
  if(value <= seil_cuda_numerical_breakdown)
    return static_cast<termination>(value);
  return termination::numerical_breakdown;
}

}

operation_result<solve_result> solve_cuda_checked(
  const prepared_problem& value,
  const common_options& common,
  const native_cuda_cold_options& options
)
{
  const auto started = std::chrono::steady_clock::now();
  if(!valid_options(common, options))
    return std::unexpected(operation_error{operation_code::invalid_argument,
      operation_stage::validation, 0, "CUDA solver options are invalid"});
  const auto* storage = detail::prepared_problem_access::get(value);
  if(storage == nullptr)
    return std::unexpected(operation_error{operation_code::invalid_argument,
      operation_stage::validation, 0, "problem is not prepared"});
  if(storage->fields && storage->fields->device != options.device_index)
    return std::unexpected(operation_error{operation_code::invalid_argument,
      operation_stage::validation, 0,
      "borrowed CUDA fields are not resident on the selected device"});

  int32_t device_count = 0;
  int status = seil_cuda_device_count(&device_count);
  if(status != 0)
    return std::unexpected(cuda_error(operation_stage::device_query, status,
      "CUDA device query failed"));
  if(options.device_index >= device_count)
    return std::unexpected(operation_error{operation_code::backend_unavailable,
      operation_stage::device_query, 0, "requested CUDA device is unavailable"});

  const auto preparation_started = std::chrono::steady_clock::now();
  auto packed = detail::pack_cuda_problem(value);
  if(!packed)
    return std::unexpected(std::move(packed.error()));
  const double preparation_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - preparation_started).count();
  const seil_cuda_options cuda_options = pack_options(*storage, common, options);
  seil_cuda_context* raw_context = nullptr;
  status = seil_cuda_create(&cuda_options, &raw_context);
  if(status != 0)
    return std::unexpected(cuda_error(operation_stage::context_create, status,
      "CUDA context creation failed"));
  owned_context context(raw_context);

  const auto upload_started = std::chrono::steady_clock::now();
  status = seil_cuda_upload_problem(context.get(), &packed->view);
  if(status != 0)
    return std::unexpected(cuda_error(operation_stage::upload, status,
      "CUDA snapshot upload failed"));
  const double upload_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - upload_started).count();

  seil_cuda_summary summary{};
  status = seil_cuda_solve_stage(context.get(), &summary);
  if(status != 0)
    return std::unexpected(cuda_error(operation_stage::solve, status,
      "CUDA solve failed"));

  size_t pose_count = 0;
  status = seil_cuda_download_poses(context.get(), nullptr, 0, &pose_count);
  if(status != 0)
    return std::unexpected(cuda_error(operation_stage::pose_count, status,
      "CUDA pose count failed"));
  if(pose_count != storage->value.nodes.size())
    return std::unexpected(operation_error{operation_code::backend_failure,
      operation_stage::pose_count, 0, "CUDA pose count does not match the problem"});
  std::vector<seil_cuda_pose4> downloaded_poses(pose_count);
  status = seil_cuda_download_poses(context.get(), downloaded_poses.data(),
    downloaded_poses.size(), &pose_count);
  if(status != 0)
    return std::unexpected(cuda_error(operation_stage::pose_download, status,
      "CUDA pose download failed"));

  size_t iteration_count = 0;
  status = seil_cuda_download_iterations(context.get(), nullptr, 0, &iteration_count);
  if(status != 0)
    return std::unexpected(cuda_error(operation_stage::iteration_count, status,
      "CUDA iteration count failed"));
  std::vector<seil_cuda_iteration> downloaded_iterations(iteration_count);
  if(iteration_count != 0) {
    status = seil_cuda_download_iterations(context.get(),
      downloaded_iterations.data(), downloaded_iterations.size(), &iteration_count);
    if(status != 0)
      return std::unexpected(cuda_error(operation_stage::iteration_download, status,
        "CUDA iteration download failed"));
  }

  solve_result result;
  result.selected_backend = backend::native_cuda_cold;
  result.stop = map_termination(summary.termination);
  result.solution_usable = summary.solution_usable != 0;
  result.initial_cost = {summary.initial_relative_cost,
    summary.initial_absolute_cost, summary.initial_registration_cost};
  result.final_cost = {summary.final_relative_cost,
    summary.final_absolute_cost, summary.final_registration_cost};
  result.registration.accepted_rows = summary.registration_rows;
  std::copy(std::begin(summary.registration_rejections),
    std::end(summary.registration_rejections), result.registration.rejections.begin());
  result.poses.reserve(downloaded_poses.size());
  for(const auto& pose : downloaded_poses)
    result.poses.push_back({pose.x, pose.y, pose.z, wrap_yaw(pose.yaw)});
  result.iterations.reserve(downloaded_iterations.size());
  for(const auto& item : downloaded_iterations)
    result.iterations.push_back({item.iteration, item.linear_iterations,
      item.accepted != 0, item.objective_before, item.objective_after,
      item.damping, item.damping > 0.0 ? 1.0 / std::sqrt(item.damping) : 0.0,
      item.gradient_norm, item.update_norm, item.iteration_seconds,
      item.linear_solver_seconds});
  result.telemetry.preparation_ms = preparation_ms;
  result.telemetry.upload_ms = upload_ms;
  result.telemetry.evaluation_ms = summary.evaluation_ms;
  result.telemetry.assembly_ms = summary.assembly_ms;
  result.telemetry.linear_solver_ms = summary.linear_solver_ms;
  result.telemetry.update_ms = summary.update_ms;
  result.telemetry.synchronization_ms = summary.synchronization_ms;
  result.telemetry.download_ms = summary.download_ms;
  result.telemetry.host_to_device_bytes = summary.host_to_device_bytes;
  result.telemetry.device_to_host_bytes = summary.device_to_host_bytes;
  result.telemetry.peak_device_bytes = summary.peak_device_bytes;
  result.telemetry.synchronization_count = summary.synchronization_count;
  result.telemetry.accepted_steps = summary.successful_steps;
  result.telemetry.rejected_steps = summary.unsuccessful_steps;
  result.telemetry.implementation = "seil_native_cuda_cold";
  result.telemetry.device = "cuda_device_" + std::to_string(options.device_index);
  result.telemetry.total_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - started).count();
  result.message = "CUDA cold-snapshot solve completed";
  return result;
}

}
