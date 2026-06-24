#include "cuda_abi.h"
#include "field_views.cuh"

#include <cuda_runtime.h>

#include <new>
#include <string>

namespace {

thread_local std::string last_error;

int fail(int status, const char* message)
{
  last_error = message;
  return status;
}

int cuda_fail(cudaError_t status)
{
  last_error = cudaGetErrorString(status);
  return 5;
}

template<class value_type>
cudaError_t upload_array(
  const value_type* source,
  std::size_t count,
  value_type** destination
)
{
  *destination = nullptr;
  if(count == 0)
    return cudaSuccess;
  cudaError_t status = cudaMalloc(reinterpret_cast<void**>(destination),
    count * sizeof(value_type));
  if(status != cudaSuccess)
    return status;
  return cudaMemcpy(*destination, source, count * sizeof(value_type),
    cudaMemcpyHostToDevice);
}

__device__ double robust_cost(seil_cuda_loss robust_loss, double residual)
{
  const double absolute = fabs(residual);
  if(robust_loss.kind == seil_cuda_loss_huber)
    return absolute <= robust_loss.scale ? 0.5 * residual * residual :
      robust_loss.scale * (absolute - 0.5 * robust_loss.scale);
  if(robust_loss.kind == seil_cuda_loss_cauchy) {
    const double normalized = residual / robust_loss.scale;
    return 0.5 * robust_loss.scale * robust_loss.scale *
      log1p(normalized * normalized);
  }
  return 0.5 * residual * residual;
}

__device__ double edge_cost(
  const seil_cuda_node* nodes,
  const seil_cuda_edge& edge
)
{
  const seil_cuda_pose4& left = nodes[edge.node_i].pose;
  const seil_cuda_pose4& right = nodes[edge.node_j].pose;
  const double cosine = cos(left.yaw);
  const double sine = sin(left.yaw);
  const double dx = right.x - left.x;
  const double dy = right.y - left.y;
  const double raw[4]{
    cosine * dx + sine * dy - edge.measured.x,
    -sine * dx + cosine * dy - edge.measured.y,
    right.z - left.z - edge.measured.z,
    atan2(sin(right.yaw - left.yaw - edge.measured.yaw),
      cos(right.yaw - left.yaw - edge.measured.yaw))
  };
  double squared = 0.0;
  for(int row = 0; row < 4; ++row) {
    double residual = 0.0;
    for(int column = 0; column < 4; ++column)
      residual += edge.sqrt_information[row * 4 + column] * raw[column];
    squared += residual * residual;
  }
  return robust_cost(edge.robust_loss, sqrt(fmax(0.0, squared)));
}

struct registration_evaluation {
  double residual;
  double confidence;
  double jacobian_i[4];
  double jacobian_j[4];
  bool valid;
};

__device__ registration_evaluation evaluate_registration(
  const seil_cuda_options& options,
  const seil_cuda_node* nodes,
  const seil_cuda_registration_group& group,
  const seil_cuda_registration_sample& sample,
  const seil_cuda_volume* volumes,
  const seil_cuda_tsdf_tile* tsdf_tiles,
  size_t tsdf_count,
  const seil_cuda_esdf_tile* esdf_tiles,
  size_t esdf_count
)
{
  registration_evaluation output{};
  const seil_cuda_pose4& reference = nodes[group.node_i].pose;
  const seil_cuda_pose4& reading = nodes[group.node_j].pose;
  const double reference_cosine = cos(reference.yaw);
  const double reference_sine = sin(reference.yaw);
  const double rotated_x = reference_cosine * sample.reference_x -
    reference_sine * sample.reference_y;
  const double rotated_y = reference_sine * sample.reference_x +
    reference_cosine * sample.reference_y;
  const double world_x = reference.x + rotated_x - reading.x;
  const double world_y = reference.y + rotated_y - reading.y;
  const double reading_cosine = cos(reading.yaw);
  const double reading_sine = sin(reading.yaw);
  const double point_x = reading_cosine * world_x + reading_sine * world_y;
  const double point_y = -reading_sine * world_x + reading_cosine * world_y;
  const double point_z = reference.z + sample.reference_z - reading.z;
  const seil_cuda_volume& volume = volumes[group.volume_j];
  seil_cuda_field_query query{};
  if(!seil_query_trilinear(group.field, group.volume_j, volume.voxel_size,
    point_x, point_y, point_z, tsdf_tiles, tsdf_count, esdf_tiles, esdf_count,
    &query))
    return output;
  seil_cuda_field_query confidence{};
  if(options.registration_profile == 0 &&
    !seil_query_trilinear(seil_cuda_field_tsdf, group.volume_j,
      volume.voxel_size, point_x, point_y, point_z, tsdf_tiles, tsdf_count,
      esdf_tiles, esdf_count, &confidence))
    return output;
  const double gx = query.gradient[0];
  const double gy = query.gradient[1];
  const double gz = query.gradient[2];
  const double gradient_norm = sqrt(gx * gx + gy * gy + gz * gz);
  if(options.registration_profile == 0 &&
    gradient_norm < options.minimum_gradient_norm_registration)
    return output;
  output.residual = sample.reference_distance - query.distance;
  if(options.maximum_absolute_residual_m > 0.0 &&
    fabs(output.residual) > options.maximum_absolute_residual_m)
    return output;
  output.confidence = options.registration_profile == 1 ? 1.0 :
    fmin(options.maximum_confidence_weight,
      fmin(sample.reference_tsdf_weight, confidence.weight));
  if(!(output.confidence > 0.0))
    return output;
  const double yaw_world_x = -rotated_y;
  const double yaw_world_y = rotated_x;
  const double yaw_reading_x = reading_cosine * yaw_world_x +
    reading_sine * yaw_world_y;
  const double yaw_reading_y = -reading_sine * yaw_world_x +
    reading_cosine * yaw_world_y;
  output.jacobian_i[0] = -(gx * reading_cosine - gy * reading_sine);
  output.jacobian_i[1] = -(gx * reading_sine + gy * reading_cosine);
  output.jacobian_i[2] = -gz;
  output.jacobian_i[3] = -(gx * yaw_reading_x + gy * yaw_reading_y);
  output.jacobian_j[0] = -output.jacobian_i[0];
  output.jacobian_j[1] = -output.jacobian_i[1];
  output.jacobian_j[2] = -output.jacobian_i[2];
  output.jacobian_j[3] = -(gx * point_y - gy * point_x);
  output.valid = true;
  return output;
}

__global__ void evaluate_deterministic(
  seil_cuda_options options,
  const seil_cuda_node* nodes,
  const seil_cuda_edge* relative_edges,
  size_t relative_count,
  const seil_cuda_edge* absolute_edges,
  size_t absolute_count,
  const seil_cuda_registration_group* groups,
  size_t group_count,
  const seil_cuda_registration_sample* samples,
  const seil_cuda_volume* volumes,
  const seil_cuda_tsdf_tile* tsdf_tiles,
  size_t tsdf_count,
  const seil_cuda_esdf_tile* esdf_tiles,
  size_t esdf_count,
  double* costs,
  uint64_t* accepted_rows
)
{
  if(blockIdx.x != 0 || threadIdx.x != 0)
    return;
  costs[0] = 0.0;
  costs[1] = 0.0;
  costs[2] = 0.0;
  *accepted_rows = 0;
  for(size_t index = 0; index < relative_count; ++index)
    costs[0] += edge_cost(nodes, relative_edges[index]);
  for(size_t index = 0; index < absolute_count; ++index)
    costs[1] += edge_cost(nodes, absolute_edges[index]);
  for(size_t group_index = 0; group_index < group_count; ++group_index) {
    const seil_cuda_registration_group& group = groups[group_index];
    double total_confidence = 0.0;
    for(uint32_t local = 0; local < group.sample_count; ++local) {
      const auto row = evaluate_registration(options, nodes, group,
        samples[group.sample_begin + local], volumes, tsdf_tiles, tsdf_count,
        esdf_tiles, esdf_count);
      if(row.valid)
        total_confidence += row.confidence;
    }
    for(uint32_t local = 0; local < group.sample_count; ++local) {
      const auto row = evaluate_registration(options, nodes, group,
        samples[group.sample_begin + local], volumes, tsdf_tiles, tsdf_count,
        esdf_tiles, esdf_count);
      if(!row.valid)
        continue;
      const double confidence = options.registration_profile == 1 ?
        row.confidence : row.confidence / total_confidence;
      costs[2] += group.direction_weight * confidence *
        robust_cost(options.registration_loss, row.residual);
      ++*accepted_rows;
    }
  }
}

__global__ void evaluate_edges_atomic(
  const seil_cuda_node* nodes,
  const seil_cuda_edge* edges,
  size_t count,
  double* cost
)
{
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < count)
    atomicAdd(cost, edge_cost(nodes, edges[index]));
}

__device__ double wrap_yaw_device(double yaw)
{
  return atan2(sin(yaw), cos(yaw));
}

__device__ seil_cuda_pose4 compose_device(
  const seil_cuda_pose4& left,
  const seil_cuda_pose4& right
)
{
  const double cosine = cos(left.yaw);
  const double sine = sin(left.yaw);
  return {
    left.x + cosine * right.x - sine * right.y,
    left.y + sine * right.x + cosine * right.y,
    left.z + right.z,
    wrap_yaw_device(left.yaw + right.yaw)
  };
}

__device__ seil_cuda_pose4 inverse_device(const seil_cuda_pose4& value)
{
  const double cosine = cos(value.yaw);
  const double sine = sin(value.yaw);
  return {
    -cosine * value.x - sine * value.y,
    sine * value.x - cosine * value.y,
    -value.z,
    wrap_yaw_device(-value.yaw)
  };
}

__device__ double relax_edge(
  seil_cuda_node* nodes,
  const seil_cuda_edge& edge,
  double maximum_translation,
  double maximum_yaw
)
{
  seil_cuda_node& left = nodes[edge.node_i];
  seil_cuda_node& right = nodes[edge.node_j];
  const seil_cuda_pose4 target_right = compose_device(left.pose, edge.measured);
  const seil_cuda_pose4 target_left = compose_device(
    right.pose, inverse_device(edge.measured));
  seil_cuda_pose4* pose = nullptr;
  seil_cuda_pose4 target{};
  double fraction = 1.0;
  if(left.active_index < 0 && right.active_index >= 0) {
    pose = &right.pose;
    target = target_right;
  } else if(right.active_index < 0 && left.active_index >= 0) {
    pose = &left.pose;
    target = target_left;
  } else if(left.active_index >= 0 && right.active_index >= 0) {
    pose = &right.pose;
    target = target_right;
    fraction = 0.5;
  } else {
    return 0.0;
  }
  double dx = fraction * (target.x - pose->x);
  double dy = fraction * (target.y - pose->y);
  double dz = fraction * (target.z - pose->z);
  double dyaw = fraction * wrap_yaw_device(target.yaw - pose->yaw);
  const double translation = sqrt(dx * dx + dy * dy + dz * dz);
  if(translation > maximum_translation) {
    const double scale = maximum_translation / translation;
    dx *= scale;
    dy *= scale;
    dz *= scale;
  }
  dyaw = fmax(-maximum_yaw, fmin(maximum_yaw, dyaw));
  pose->x += dx;
  pose->y += dy;
  pose->z += dz;
  pose->yaw = wrap_yaw_device(pose->yaw + dyaw);
  return sqrt(dx * dx + dy * dy + dz * dz + dyaw * dyaw);
}

__global__ void solve_sparse_system_serial(
  seil_cuda_options options,
  seil_cuda_node* nodes,
  const seil_cuda_edge* relative_edges,
  size_t relative_count,
  const seil_cuda_edge* absolute_edges,
  size_t absolute_count,
  uint32_t* completed_iterations,
  double* update_norm
)
{
  if(blockIdx.x != 0 || threadIdx.x != 0)
    return;
  *completed_iterations = 0;
  *update_norm = 0.0;
  for(uint32_t iteration = 0; iteration < options.maximum_iterations; ++iteration) {
    double largest_update = 0.0;
    for(size_t index = 0; index < relative_count; ++index)
      largest_update = fmax(largest_update, relax_edge(nodes, relative_edges[index],
        options.maximum_translation_step_m, options.maximum_yaw_step_rad));
    for(size_t index = 0; index < absolute_count; ++index)
      largest_update = fmax(largest_update, relax_edge(nodes, absolute_edges[index],
        options.maximum_translation_step_m, options.maximum_yaw_step_rad));
    *completed_iterations = iteration + 1;
    *update_norm = largest_update;
    if(largest_update <= options.minimum_update_norm)
      return;
  }
}

}

struct seil_cuda_compact_state {
  double costs[6];
  uint64_t accepted_rows;
  uint32_t completed_iterations;
  double update_norm;
};

struct seil_cuda_context {
  seil_cuda_options options{};
  seil_cuda_node* nodes{};
  seil_cuda_edge* relative_edges{};
  seil_cuda_edge* absolute_edges{};
  seil_cuda_registration_group* registration_groups{};
  seil_cuda_registration_sample* registration_samples{};
  seil_cuda_volume* volumes{};
  seil_cuda_tsdf_tile* tsdf_tiles{};
  seil_cuda_esdf_tile* esdf_tiles{};
  seil_cuda_sparse_slot* sparse_slots{};
  seil_cuda_compact_state* compact{};
  int32_t active_node_count{};
  std::size_t node_count{};
  std::size_t relative_edge_count{};
  std::size_t absolute_edge_count{};
  std::size_t registration_group_count{};
  std::size_t registration_sample_count{};
  std::size_t volume_count{};
  std::size_t tsdf_tile_count{};
  std::size_t esdf_tile_count{};
  std::size_t sparse_slot_count{};
  std::uint64_t uploaded_bytes{};
  seil_cuda_iteration iteration{};
  bool solved{};
};

extern "C" int seil_cuda_abi_version(void)
{
  return seil_cuda_abi_version_value;
}

extern "C" const char* seil_cuda_last_error(void)
{
  return last_error.c_str();
}

extern "C" int seil_cuda_device_count(int32_t* output_count)
{
  if(output_count == nullptr)
    return fail(3, "device count output is null");
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  if(status != cudaSuccess)
    return cuda_fail(status);
  *output_count = count;
  return 0;
}

extern "C" int seil_cuda_create(
  const seil_cuda_options* options,
  seil_cuda_context** output_context
)
{
  if(options == nullptr || output_context == nullptr)
    return fail(3, "CUDA create argument is null");
  *output_context = nullptr;
  if(options->abi_version != seil_cuda_abi_version_value ||
    options->struct_size != sizeof(seil_cuda_options))
    return fail(2, "CUDA options ABI does not match");
  cudaError_t status = cudaSetDevice(options->device_index);
  if(status != cudaSuccess)
    return cuda_fail(status);
  auto* context = new (std::nothrow) seil_cuda_context;
  if(context == nullptr)
    return fail(4, "CUDA context allocation failed");
  context->options = *options;
  *output_context = context;
  return 0;
}

extern "C" int seil_cuda_upload_problem(
  seil_cuda_context* context,
  const seil_cuda_problem* problem
)
{
  if(context == nullptr || problem == nullptr)
    return fail(3, "CUDA upload argument is null");
  if(problem->abi_version != seil_cuda_abi_version_value ||
    problem->struct_size != sizeof(seil_cuda_problem))
    return fail(2, "CUDA problem ABI does not match");
  const std::uint64_t requested =
    problem->node_count * sizeof(seil_cuda_node) +
    problem->relative_edge_count * sizeof(seil_cuda_edge) +
    problem->absolute_edge_count * sizeof(seil_cuda_edge) +
    problem->registration_group_count * sizeof(seil_cuda_registration_group) +
    problem->registration_sample_count * sizeof(seil_cuda_registration_sample) +
    problem->volume_count * sizeof(seil_cuda_volume) +
    problem->tsdf_tile_count * sizeof(seil_cuda_tsdf_tile) +
    problem->esdf_tile_count * sizeof(seil_cuda_esdf_tile) +
    problem->sparse_slot_count * sizeof(seil_cuda_sparse_slot);
  if(context->options.memory_budget_bytes != 0 &&
    requested > context->options.memory_budget_bytes)
    return fail(4, "CUDA memory budget is insufficient");

  cudaError_t status = upload_array(problem->nodes, problem->node_count,
    &context->nodes);
  if(status == cudaSuccess)
    status = upload_array(problem->relative_edges, problem->relative_edge_count,
      &context->relative_edges);
  if(status == cudaSuccess)
    status = upload_array(problem->absolute_edges, problem->absolute_edge_count,
      &context->absolute_edges);
  if(status == cudaSuccess)
    status = upload_array(problem->registration_groups,
      problem->registration_group_count, &context->registration_groups);
  if(status == cudaSuccess)
    status = upload_array(problem->registration_samples,
      problem->registration_sample_count, &context->registration_samples);
  if(status == cudaSuccess)
    status = upload_array(problem->volumes, problem->volume_count,
      &context->volumes);
  if(status == cudaSuccess)
    status = upload_array(problem->tsdf_tiles, problem->tsdf_tile_count,
      &context->tsdf_tiles);
  if(status == cudaSuccess)
    status = upload_array(problem->esdf_tiles, problem->esdf_tile_count,
      &context->esdf_tiles);
  if(status == cudaSuccess)
    status = upload_array(problem->sparse_slots, problem->sparse_slot_count,
      &context->sparse_slots);
  if(status == cudaSuccess)
    status = cudaMalloc(reinterpret_cast<void**>(&context->compact),
      sizeof(seil_cuda_compact_state));
  if(status != cudaSuccess)
    return cuda_fail(status);
  context->node_count = problem->node_count;
  context->active_node_count = problem->active_node_count;
  context->relative_edge_count = problem->relative_edge_count;
  context->absolute_edge_count = problem->absolute_edge_count;
  context->registration_group_count = problem->registration_group_count;
  context->registration_sample_count = problem->registration_sample_count;
  context->volume_count = problem->volume_count;
  context->tsdf_tile_count = problem->tsdf_tile_count;
  context->esdf_tile_count = problem->esdf_tile_count;
  context->sparse_slot_count = problem->sparse_slot_count;
  context->uploaded_bytes = requested;
  return 0;
}

extern "C" int seil_cuda_solve_stage(
  seil_cuda_context* context,
  seil_cuda_summary* output_summary
)
{
  if(context == nullptr || output_summary == nullptr)
    return fail(3, "CUDA solve argument is null");
  *output_summary = {};
  cudaError_t status = cudaMemset(context->compact, 0,
    sizeof(seil_cuda_compact_state));
  if(status != cudaSuccess)
    return cuda_fail(status);
  auto* compact_bytes = reinterpret_cast<unsigned char*>(context->compact);
  auto* initial_costs = reinterpret_cast<double*>(compact_bytes);
  auto* final_costs = initial_costs + 3;
  auto* accepted_rows = reinterpret_cast<uint64_t*>(
    compact_bytes + offsetof(seil_cuda_compact_state, accepted_rows));
  auto* completed_iterations = reinterpret_cast<uint32_t*>(
    compact_bytes + offsetof(seil_cuda_compact_state, completed_iterations));
  auto* update_norm = reinterpret_cast<double*>(
    compact_bytes + offsetof(seil_cuda_compact_state, update_norm));
  const auto launch_evaluation = [&](double* costs) {
    if(context->options.reduction_policy == seil_cuda_reduction_atomic_fast) {
      constexpr unsigned threads = 128;
      if(context->relative_edge_count != 0)
        evaluate_edges_atomic<<<
          static_cast<unsigned>((context->relative_edge_count + threads - 1) / threads),
          threads>>>(context->nodes, context->relative_edges,
            context->relative_edge_count, costs);
      if(context->absolute_edge_count != 0)
        evaluate_edges_atomic<<<
          static_cast<unsigned>((context->absolute_edge_count + threads - 1) / threads),
          threads>>>(context->nodes, context->absolute_edges,
            context->absolute_edge_count, costs + 1);
    } else {
      evaluate_deterministic<<<1, 1>>>(context->options, context->nodes,
        context->relative_edges, context->relative_edge_count,
        context->absolute_edges, context->absolute_edge_count,
        context->registration_groups, context->registration_group_count,
        context->registration_samples, context->volumes,
        context->tsdf_tiles, context->tsdf_tile_count,
        context->esdf_tiles, context->esdf_tile_count,
        costs, accepted_rows);
    }
  };
  launch_evaluation(initial_costs);
  if(context->active_node_count != 0)
    solve_sparse_system_serial<<<1, 1>>>(context->options, context->nodes,
      context->relative_edges, context->relative_edge_count,
      context->absolute_edges, context->absolute_edge_count,
      completed_iterations, update_norm);
  else {
    status = cudaMemset(completed_iterations, 0, sizeof(uint32_t));
    if(status == cudaSuccess)
      status = cudaMemset(update_norm, 0, sizeof(double));
  }
  if(status != cudaSuccess)
    return cuda_fail(status);
  launch_evaluation(final_costs);
  seil_cuda_compact_state compact{};
  status = cudaMemcpy(&compact, context->compact, sizeof(compact),
    cudaMemcpyDeviceToHost);
  if(status != cudaSuccess)
    return cuda_fail(status);
  output_summary->initial_relative_cost = compact.costs[0];
  output_summary->initial_absolute_cost = compact.costs[1];
  output_summary->initial_registration_cost = compact.costs[2];
  output_summary->final_relative_cost = compact.costs[3];
  output_summary->final_absolute_cost = compact.costs[4];
  output_summary->final_registration_cost = compact.costs[5];
  output_summary->registration_rows = static_cast<uint32_t>(compact.accepted_rows);
  output_summary->termination = context->active_node_count == 0 ?
    seil_cuda_no_free_nodes : seil_cuda_converged_update;
  output_summary->solution_usable = 1;
  output_summary->successful_steps = compact.completed_iterations == 0 ? 0 : 1;
  context->iteration = {
    compact.completed_iterations == 0 ? 0U : compact.completed_iterations - 1U,
    compact.completed_iterations,
    1U,
    output_summary->initial_relative_cost + output_summary->initial_absolute_cost +
      output_summary->initial_registration_cost,
    output_summary->final_relative_cost + output_summary->final_absolute_cost +
      output_summary->final_registration_cost,
    context->options.initial_damping,
    0.0,
    compact.update_norm,
    0.0,
    0.0
  };
  context->solved = true;
  output_summary->requested_device_bytes = context->uploaded_bytes +
    sizeof(seil_cuda_compact_state);
  output_summary->peak_device_bytes = output_summary->requested_device_bytes;
  output_summary->host_to_device_bytes = context->uploaded_bytes;
  output_summary->device_to_host_bytes = sizeof(seil_cuda_compact_state);
  output_summary->synchronization_count = 1;
  return 0;
}

extern "C" int seil_cuda_download_poses(
  seil_cuda_context* context,
  seil_cuda_pose4* output,
  size_t capacity,
  size_t* output_count
)
{
  if(context == nullptr || output_count == nullptr)
    return fail(3, "CUDA pose download argument is null");
  *output_count = context->node_count;
  if(output == nullptr)
    return 0;
  if(capacity < context->node_count)
    return fail(3, "CUDA pose download capacity is insufficient");
  auto* nodes = new (std::nothrow) seil_cuda_node[context->node_count];
  if(nodes == nullptr)
    return fail(4, "CUDA pose staging allocation failed");
  const cudaError_t status = cudaMemcpy(nodes, context->nodes,
    context->node_count * sizeof(seil_cuda_node), cudaMemcpyDeviceToHost);
  if(status == cudaSuccess)
    for(size_t index = 0; index < context->node_count; ++index)
      output[index] = nodes[index].pose;
  delete[] nodes;
  return status == cudaSuccess ? 0 : cuda_fail(status);
}

extern "C" int seil_cuda_download_iterations(
  seil_cuda_context* context,
  seil_cuda_iteration* output,
  size_t capacity,
  size_t* output_count
)
{
  if(context == nullptr || output_count == nullptr)
    return fail(3, "CUDA iteration download argument is null");
  *output_count = context->solved && context->active_node_count != 0 ? 1U : 0U;
  if(output == nullptr || *output_count == 0)
    return 0;
  if(capacity < 1)
    return fail(3, "CUDA iteration download capacity is insufficient");
  output[0] = context->iteration;
  return 0;
}

extern "C" int seil_cuda_destroy(seil_cuda_context* context)
{
  if(context == nullptr)
    return 0;
  cudaFree(context->nodes);
  cudaFree(context->relative_edges);
  cudaFree(context->absolute_edges);
  cudaFree(context->registration_groups);
  cudaFree(context->registration_samples);
  cudaFree(context->volumes);
  cudaFree(context->tsdf_tiles);
  cudaFree(context->esdf_tiles);
  cudaFree(context->sparse_slots);
  cudaFree(context->compact);
  delete context;
  return 0;
}
