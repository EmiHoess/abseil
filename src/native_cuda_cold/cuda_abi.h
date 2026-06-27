#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define abseil_cuda_export __declspec(dllexport)
#else
#define abseil_cuda_export __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum { abseil_cuda_abi_version_value = 1 };
enum { abseil_cuda_tile_edge = 8 };
enum { abseil_cuda_tile_voxels = 512 };

typedef enum abseil_cuda_loss_kind {
  abseil_cuda_loss_none = 0,
  abseil_cuda_loss_huber = 1,
  abseil_cuda_loss_cauchy = 2
} abseil_cuda_loss_kind;

typedef enum abseil_cuda_field_kind {
  abseil_cuda_field_tsdf = 0,
  abseil_cuda_field_esdf = 1
} abseil_cuda_field_kind;

typedef enum abseil_cuda_reduction_policy {
  abseil_cuda_reduction_deterministic = 0,
  abseil_cuda_reduction_atomic_fast = 1
} abseil_cuda_reduction_policy;

typedef enum abseil_cuda_termination {
  abseil_cuda_no_free_nodes = 0,
  abseil_cuda_converged_cost = 1,
  abseil_cuda_converged_gradient = 2,
  abseil_cuda_converged_update = 3,
  abseil_cuda_maximum_iterations = 4,
  abseil_cuda_maximum_time = 5,
  abseil_cuda_rejection_limit = 6,
  abseil_cuda_singular_graph = 7,
  abseil_cuda_numerical_breakdown = 8
} abseil_cuda_termination;

typedef struct abseil_cuda_pose4 {
  double x;
  double y;
  double z;
  double yaw;
} abseil_cuda_pose4;

typedef struct abseil_cuda_loss {
  uint32_t kind;
  double scale;
} abseil_cuda_loss;

typedef struct abseil_cuda_node {
  abseil_cuda_pose4 pose;
  int32_t active_index;
} abseil_cuda_node;

typedef struct abseil_cuda_edge {
  int32_t node_i;
  int32_t node_j;
  int32_t slot_ii;
  int32_t slot_ij;
  int32_t slot_jj;
  uint32_t transpose_ij;
  abseil_cuda_pose4 measured;
  double sqrt_information[16];
  abseil_cuda_loss robust_loss;
} abseil_cuda_edge;

typedef struct abseil_cuda_registration_group {
  int32_t node_i;
  int32_t node_j;
  int32_t volume_i;
  int32_t volume_j;
  int32_t slot_ii;
  int32_t slot_ij;
  int32_t slot_jj;
  uint32_t transpose_ij;
  uint32_t field;
  uint32_t sample_begin;
  uint32_t sample_count;
  double direction_weight;
} abseil_cuda_registration_group;

typedef struct abseil_cuda_registration_sample {
  uint32_t group_index;
  double reference_x;
  double reference_y;
  double reference_z;
  double reference_distance;
  double reference_tsdf_weight;
} abseil_cuda_registration_sample;

typedef struct abseil_cuda_volume {
  int32_t node_index;
  double voxel_size;
  uint32_t has_tsdf;
  uint32_t has_esdf;
} abseil_cuda_volume;

typedef struct abseil_cuda_tsdf_voxel {
  float distance;
  float weight;
  uint32_t flags;
} abseil_cuda_tsdf_voxel;

typedef struct abseil_cuda_tsdf_tile {
  int32_t volume_index;
  int32_t x;
  int32_t y;
  int32_t z;
  abseil_cuda_tsdf_voxel voxels[abseil_cuda_tile_voxels];
} abseil_cuda_tsdf_tile;

typedef struct abseil_cuda_esdf_voxel {
  float distance;
  uint32_t flags;
} abseil_cuda_esdf_voxel;

typedef struct abseil_cuda_esdf_tile {
  int32_t volume_index;
  int32_t x;
  int32_t y;
  int32_t z;
  abseil_cuda_esdf_voxel voxels[abseil_cuda_tile_voxels];
} abseil_cuda_esdf_tile;

typedef struct abseil_cuda_field_tile {
  int32_t volume_index;
  int32_t x;
  int32_t y;
  int32_t z;
  const unsigned char* distance;
  const unsigned char* weight;
  const unsigned char* flags;
  size_t voxel_stride;
  uint32_t encoding;
  float voxel_size;
  float maximum_distance;
} abseil_cuda_field_tile;

typedef struct abseil_cuda_sparse_slot {
  int32_t block_row;
  int32_t block_column;
} abseil_cuda_sparse_slot;

typedef struct abseil_cuda_problem {
  uint32_t abi_version;
  uint32_t struct_size;
  const abseil_cuda_node* nodes;
  size_t node_count;
  int32_t active_node_count;
  const abseil_cuda_edge* relative_edges;
  size_t relative_edge_count;
  const abseil_cuda_edge* absolute_edges;
  size_t absolute_edge_count;
  const abseil_cuda_registration_group* registration_groups;
  size_t registration_group_count;
  const abseil_cuda_registration_sample* registration_samples;
  size_t registration_sample_count;
  const abseil_cuda_volume* volumes;
  size_t volume_count;
  const abseil_cuda_tsdf_tile* tsdf_tiles;
  size_t tsdf_tile_count;
  const abseil_cuda_esdf_tile* esdf_tiles;
  size_t esdf_tile_count;
  const abseil_cuda_field_tile* resident_tsdf;
  size_t resident_tsdf_count;
  const abseil_cuda_field_tile* resident_esdf;
  size_t resident_esdf_count;
  const abseil_cuda_sparse_slot* sparse_slots;
  size_t sparse_slot_count;
  const char* content_hash;
} abseil_cuda_problem;

typedef struct abseil_cuda_options {
  uint32_t abi_version;
  uint32_t struct_size;
  int32_t device_index;
  uint64_t memory_budget_bytes;
  uint32_t maximum_iterations;
  uint32_t maximum_rejected_steps;
  uint32_t pcg_maximum_iterations;
  double maximum_seconds;
  double initial_damping;
  double minimum_damping;
  double maximum_damping;
  double minimum_cost_change;
  double minimum_gradient_norm;
  double minimum_update_norm;
  double maximum_translation_step_m;
  double maximum_yaw_step_rad;
  double pcg_tolerance;
  double minimum_tsdf_weight;
  double maximum_confidence_weight;
  double minimum_gradient_norm_registration;
  double maximum_absolute_residual_m;
  uint32_t allow_nearest_fallback;
  uint32_t reduction_policy;
  uint32_t registration_profile;
  abseil_cuda_loss registration_loss;
} abseil_cuda_options;

typedef struct abseil_cuda_iteration {
  uint32_t iteration;
  uint32_t linear_iterations;
  uint32_t accepted;
  double objective_before;
  double objective_after;
  double damping;
  double gradient_norm;
  double update_norm;
  double iteration_seconds;
  double linear_solver_seconds;
} abseil_cuda_iteration;

typedef struct abseil_cuda_summary {
  uint32_t termination;
  uint32_t solution_usable;
  uint32_t successful_steps;
  uint32_t unsuccessful_steps;
  uint32_t registration_rows;
  uint64_t registration_rejections[9];
  double initial_relative_cost;
  double initial_absolute_cost;
  double initial_registration_cost;
  double final_relative_cost;
  double final_absolute_cost;
  double final_registration_cost;
  double upload_ms;
  double evaluation_ms;
  double assembly_ms;
  double linear_solver_ms;
  double update_ms;
  double synchronization_ms;
  double download_ms;
  double total_ms;
  uint64_t requested_device_bytes;
  uint64_t peak_device_bytes;
  uint64_t host_to_device_bytes;
  uint64_t device_to_host_bytes;
  uint64_t synchronization_count;
  int32_t runtime_version;
  int32_t driver_version;
  int32_t compute_major;
  int32_t compute_minor;
  char device_name[128];
} abseil_cuda_summary;

typedef struct abseil_cuda_context abseil_cuda_context;

abseil_cuda_export int abseil_cuda_abi_version(void);
abseil_cuda_export const char* abseil_cuda_last_error(void);
abseil_cuda_export int abseil_cuda_device_count(int32_t* output_count);
abseil_cuda_export int abseil_cuda_create(
  const abseil_cuda_options* options,
  abseil_cuda_context** output_context
);
abseil_cuda_export int abseil_cuda_upload_problem(
  abseil_cuda_context* context,
  const abseil_cuda_problem* problem
);
abseil_cuda_export int abseil_cuda_solve_stage(
  abseil_cuda_context* context,
  abseil_cuda_summary* output_summary
);
abseil_cuda_export int abseil_cuda_download_poses(
  abseil_cuda_context* context,
  abseil_cuda_pose4* output,
  size_t capacity,
  size_t* output_count
);
abseil_cuda_export int abseil_cuda_download_iterations(
  abseil_cuda_context* context,
  abseil_cuda_iteration* output,
  size_t capacity,
  size_t* output_count
);
abseil_cuda_export int abseil_cuda_destroy(abseil_cuda_context* context);

#ifdef __cplusplus
}
#endif
