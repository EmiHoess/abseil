#include "cuda_abi.h"

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

}

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
  std::size_t node_count{};
  std::uint64_t uploaded_bytes{};
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
  if(status != cudaSuccess)
    return cuda_fail(status);
  context->node_count = problem->node_count;
  context->uploaded_bytes = requested;
  return 0;
}

extern "C" int seil_cuda_solve_stage(seil_cuda_context*, seil_cuda_summary*)
{
  return fail(1, "CUDA solve stage is not available yet");
}

extern "C" int seil_cuda_download_poses(
  seil_cuda_context*,
  seil_cuda_pose4*,
  size_t,
  size_t* output_count
)
{
  if(output_count != nullptr)
    *output_count = 0;
  return fail(1, "CUDA pose download is not available yet");
}

extern "C" int seil_cuda_download_iterations(
  seil_cuda_context*,
  seil_cuda_iteration*,
  size_t,
  size_t* output_count
)
{
  if(output_count != nullptr)
    *output_count = 0;
  return fail(1, "CUDA iteration download is not available yet");
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
  delete context;
  return 0;
}
