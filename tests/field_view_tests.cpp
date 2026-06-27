#include <abseil/field.hpp>
#include <abseil/field_views.hpp>
#include <abseil/solver.hpp>

#if defined(ABSEIL_HAS_CUDA)
#include <cuda_runtime_api.h>
#endif

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>

namespace {

struct borrowed_voxel {
  float distance{};
  float weight{};
  std::uint32_t flags{};
};

void require(bool condition, const char* message)
{
  if(!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

abseil::problem field_problem()
{
  abseil::problem value;
  value.nodes.push_back({1, {}, true});
  value.volumes.push_back({0, 1.0, true, true});
  return value;
}

abseil::problem solver_problem()
{
  abseil::problem value;
  value.nodes.push_back({1, {}, true});
  value.nodes.push_back({2, {0.2, 0.0, 0.0, 0.0}, false});
  value.relative_factors.push_back({0, 1, {0.0, 0.0, 0.0, 0.0}});
  value.volumes.push_back({0, 1.0, true, false});
  value.volumes.push_back({1, 1.0, true, false});
  value.registration_samples.push_back({0, {2.5, 2.5, 2.5}, 2.0, 3.0, 0});
  value.registration_groups.push_back({0, 1, 0, 1, abseil::field_kind::tsdf,
    0, 1, 1.0, 0});
  return value;
}

std::shared_ptr<abseil::field_snapshot> snapshot(
  std::shared_ptr<std::vector<borrowed_voxel>> storage
)
{
  for(std::size_t z = 0; z < abseil::tile_edge; ++z)
    for(std::size_t y = 0; y < abseil::tile_edge; ++y)
      for(std::size_t x = 0; x < abseil::tile_edge; ++x) {
        const std::size_t index = x + 8U * (y + 8U * z);
        (*storage)[index] = {static_cast<float>(x), 3.0F, 1U};
      }
  auto fields = std::make_shared<abseil::field_snapshot>();
  fields->identity = 81;
  fields->owners.push_back(storage);
  const auto* base = reinterpret_cast<const std::byte*>(storage->data());
  fields->tiles.push_back({0, 0, 0, 0, abseil::field_kind::tsdf,
    base, base + offsetof(borrowed_voxel, weight),
    base + offsetof(borrowed_voxel, flags), sizeof(borrowed_voxel),
    abseil::field_encoding::signed_metres, 1.0F, 10.0F, 1, 2, 3});
  return fields;
}

#if defined(ABSEIL_HAS_CUDA)
void verify_cuda_snapshot(const std::vector<borrowed_voxel>& source)
{
  int device_count = 0;
  if(cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
    return;
  require(cudaSetDevice(0) == cudaSuccess, "CUDA field device selection");
  void* allocation = nullptr;
  const std::size_t bytes = source.size() * sizeof(borrowed_voxel);
  require(cudaMalloc(&allocation, bytes) == cudaSuccess,
    "CUDA field allocation");
  auto owner = std::shared_ptr<const void>(allocation, [](const void* value) {
    cudaFree(const_cast<void*>(value));
  });
  require(cudaMemcpy(allocation, source.data(), bytes,
    cudaMemcpyHostToDevice) == cudaSuccess, "CUDA field upload");

  auto fields = std::make_shared<abseil::field_snapshot>();
  fields->identity = 82;
  fields->device = 0;
  fields->owners.push_back(owner);
  const auto* base = reinterpret_cast<const std::byte*>(allocation);
  fields->tiles.push_back({1, 0, 0, 0, abseil::field_kind::tsdf,
    base, base + offsetof(borrowed_voxel, weight),
    base + offsetof(borrowed_voxel, flags), sizeof(borrowed_voxel),
    abseil::field_encoding::signed_metres, 1.0F, 10.0F, 1, 2, 3});
  auto prepared = abseil::prepare_fields(solver_problem(), fields);
  require(prepared.has_value(), "CUDA field preparation");
  abseil::native_cuda_cold_options options;
  options.device_index = 0;
  auto solved = abseil::solve_cuda_checked(*prepared, {}, options);
  require(solved.has_value() && solved->registration.accepted_rows == 1,
    "CUDA resident field read");
  options.device_index = 1;
  solved = abseil::solve_cuda_checked(*prepared, {}, options);
  require(!solved && solved.error().stage == abseil::operation_stage::validation,
    "CUDA resident device mismatch");
}
#endif

}

int main()
{
  auto storage = std::make_shared<std::vector<borrowed_voxel>>(abseil::tile_voxels);
  auto fields = snapshot(storage);
  auto prepared = abseil::prepare_fields(field_problem(), fields);
  require(prepared.has_value(), "borrowed field preparation");
  require(abseil::cpu_fields_accessible(*prepared), "host fields accessible");
  auto query = abseil::query_field(*prepared, abseil::field_kind::tsdf, 0,
    {2.2, 3.2, 4.2}, abseil::field_interpolation::nearest, false);
  require(query.has_value() && std::abs(query->sample.distance - 2.0) < 1.0e-12 &&
    query->sample.weight == 3.0, "borrowed field read");
  if(abseil::backend_available(abseil::backend::ceres_cpu)) {
    abseil::solve_options options;
    options.selected_backend = abseil::backend::ceres_cpu;
    options.specific = abseil::ceres_cpu_options{};
    auto rejected = abseil::solve(*prepared, options);
    require(!rejected && rejected.error().stage == abseil::operation_stage::validation,
      "Ceres borrowed fields rejected");
  }

  const std::string first_hash = prepared->hash();
  fields->tiles[0].content_generation = 2;
  prepared = abseil::prepare_fields(field_problem(), fields);
  require(prepared.has_value() && prepared->hash() != first_hash,
    "field generation identity");

  fields->device = 0;
  prepared = abseil::prepare_fields(field_problem(), fields);
  require(prepared.has_value() && !abseil::cpu_fields_accessible(*prepared),
    "device fields inaccessible to CPU");
  query = abseil::query_field(*prepared, abseil::field_kind::tsdf, 0,
    {2.2, 3.2, 4.2}, abseil::field_interpolation::nearest, false);
  require(!query, "device field CPU read rejected");

  auto invalid = snapshot(storage);
  invalid->tiles[0].slot_generation = 0;
  require(!abseil::validate_fields(field_problem(), *invalid),
    "invalid field generation rejected");

  auto squared = std::make_shared<std::vector<borrowed_voxel>>(abseil::tile_voxels);
  for(std::size_t index = 0; index < squared->size(); ++index) {
    const float coordinate = static_cast<float>(index % abseil::tile_edge);
    (*squared)[index] = {coordinate * coordinate, 0.0F,
      coordinate == 2.0F ? 3U : 1U};
  }
  auto squared_fields = std::make_shared<abseil::field_snapshot>();
  squared_fields->identity = 83;
  squared_fields->owners.push_back(squared);
  const auto* squared_base = reinterpret_cast<const std::byte*>(squared->data());
  squared_fields->tiles.push_back({0, 0, 0, 0, abseil::field_kind::esdf,
    squared_base, nullptr, squared_base + offsetof(borrowed_voxel, flags),
    sizeof(borrowed_voxel), abseil::field_encoding::squared_voxels,
    0.5F, 3.0F, 1, 1, 1});
  prepared = abseil::prepare_fields(field_problem(), squared_fields);
  require(prepared.has_value(), "squared field preparation");
  query = abseil::query_field(*prepared, abseil::field_kind::esdf, 0,
    {2.2, 1.2, 1.2}, abseil::field_interpolation::nearest, false);
  require(query.has_value() && query->sample.distance == -1.0,
    "squared field decoding");
  std::weak_ptr<const void> retained = squared;
  squared.reset();
  squared_fields.reset();
  require(!retained.expired(), "field owner retention");

#if defined(ABSEIL_HAS_CUDA)
  verify_cuda_snapshot(*storage);
#endif
}
