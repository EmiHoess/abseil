#include <seil/seil.hpp>

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct benchmark_config {
  seil::backend selected_backend{seil::backend::native_cpu};
  std::size_t nodes{36};
  std::size_t repetitions{7};
};

struct benchmark_record {
  double preparation_ms{};
  seil::solver_telemetry telemetry{};
  seil::cost_breakdown initial_cost{};
  seil::cost_breakdown final_cost{};
  std::size_t iterations{};
  seil::termination stop{seil::termination::numerical_breakdown};
};

seil::operation_error benchmark_error(std::string message)
{
  return {seil::operation_code::invalid_argument,
    seil::operation_stage::validation, 0, std::move(message)};
}

seil::operation_result<std::size_t> parse_count(
  std::string_view text,
  std::string_view name
)
{
  std::uint64_t parsed = 0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
  if(result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
    parsed > std::numeric_limits<std::size_t>::max())
    return std::unexpected(benchmark_error(std::string(name) + " is invalid"));
  return static_cast<std::size_t>(parsed);
}

seil::operation_result<benchmark_config> parse_config(int argc, char** argv)
{
  if(argc > 4)
    return std::unexpected(benchmark_error("too many arguments"));
  benchmark_config output;
  if(argc > 1) {
    auto selected = seil::parse_backend(argv[1]);
    if(!selected)
      return std::unexpected(std::move(selected.error()));
    output.selected_backend = *selected;
  }
  if(argc > 2) {
    auto nodes = parse_count(argv[2], "node count");
    if(!nodes)
      return std::unexpected(std::move(nodes.error()));
    output.nodes = *nodes;
  }
  if(argc > 3) {
    auto repetitions = parse_count(argv[3], "repetition count");
    if(!repetitions)
      return std::unexpected(std::move(repetitions.error()));
    output.repetitions = *repetitions;
  }
  if(output.nodes < 2)
    return std::unexpected(benchmark_error("node count must be at least two"));
  if(output.repetitions == 0)
    return std::unexpected(benchmark_error("repetition count must be positive"));
  if(!seil::backend_available(output.selected_backend))
    return std::unexpected(benchmark_error(std::string(
      seil::backend_name(output.selected_backend)) + " is unavailable"));
  return output;
}

seil::problem chain_problem(std::size_t count)
{
  seil::problem value;
  value.nodes.reserve(count);
  value.relative_factors.reserve(count);
  for(std::size_t index = 0; index < count; ++index)
    value.nodes.push_back({static_cast<std::uint64_t>(index + 1),
      {static_cast<double>(index) * 1.01,
        0.01 * std::sin(static_cast<double>(index)), 0.0,
        0.001 * static_cast<double>(index)}, index == 0});

  for(std::size_t index = 1; index < count; ++index)
    value.relative_factors.push_back({index - 1, index,
      {1.0, 0.0, 0.0, 0.0}});
  seil::relative_factor loop;
  loop.node_i = 0;
  loop.node_j = count - 1;
  loop.measured = {static_cast<double>(count - 1), 0.0, 0.0, 0.0};
  loop.sqrt_information = seil::identity_matrix4(0.25);
  value.relative_factors.push_back(loop);
  return value;
}

seil::solve_options solver_options(seil::backend selected)
{
  seil::solve_options options;
  options.selected_backend = selected;
  options.common.maximum_iterations = 50;
  switch(selected) {
    case seil::backend::native_cpu:
      options.specific = seil::native_cpu_options{};
      break;
    case seil::backend::ceres_cpu:
      options.specific = seil::ceres_cpu_options{};
      break;
    case seil::backend::native_cuda_cold:
      options.specific = seil::native_cuda_cold_options{};
      break;
  }
  return options;
}

seil::operation_result<benchmark_record> measured_run(
  const benchmark_config& config,
  std::string& hash
)
{
  const auto started = std::chrono::steady_clock::now();
  auto prepared = seil::prepare(chain_problem(config.nodes));
  if(!prepared)
    return std::unexpected(std::move(prepared.error()));
  const double preparation_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - started).count();
  if(hash.empty())
    hash = prepared->hash();
  else if(hash != prepared->hash())
    return std::unexpected(benchmark_error("prepared problem hash changed"));
  auto solved = seil::solve(*prepared, solver_options(config.selected_backend));
  if(!solved)
    return std::unexpected(std::move(solved.error()));
  if(!solved->solution_usable)
    return std::unexpected(benchmark_error("solver returned an unusable result"));
  return benchmark_record{preparation_ms, solved->telemetry,
    solved->initial_cost, solved->final_cost, solved->iterations.size(),
    solved->stop};
}

double average(const std::vector<double>& values)
{
  double total = 0.0;
  for(double value : values)
    total += value;
  return total / static_cast<double>(values.size());
}

int run(int argc, char** argv)
{
  auto config = parse_config(argc, argv);
  if(!config) {
    std::cerr << "seil_benchmark: " << config.error().message << '\n';
    return EXIT_FAILURE;
  }
  std::string hash;
  std::vector<benchmark_record> records;
  records.reserve(config->repetitions);
  for(std::size_t index = 0; index < config->repetitions; ++index) {
    auto record = measured_run(*config, hash);
    if(!record) {
      std::cerr << "seil_benchmark: " << record.error().message << '\n';
      return EXIT_FAILURE;
    }
    records.push_back(*record);
  }
  std::vector<double> preparation;
  std::vector<double> total;
  for(const auto& record : records) {
    preparation.push_back(record.preparation_ms);
    total.push_back(record.telemetry.total_ms);
  }
  const auto& last = records.back();
  std::cout << std::fixed << std::setprecision(3)
    << "backend=" << seil::backend_name(config->selected_backend)
    << " nodes=" << config->nodes
    << " repetitions=" << config->repetitions
    << " hash=" << hash
    << " preparation_ms=" << average(preparation)
    << " solve_ms=" << average(total)
    << " upload_ms=" << last.telemetry.upload_ms
    << " evaluation_ms=" << last.telemetry.evaluation_ms
    << " assembly_ms=" << last.telemetry.assembly_ms
    << " linear_ms=" << last.telemetry.linear_solver_ms
    << " update_ms=" << last.telemetry.update_ms
    << " synchronization_ms=" << last.telemetry.synchronization_ms
    << " download_ms=" << last.telemetry.download_ms
    << " h2d_bytes=" << last.telemetry.host_to_device_bytes
    << " d2h_bytes=" << last.telemetry.device_to_host_bytes
    << " peak_device_bytes=" << last.telemetry.peak_device_bytes
    << " synchronizations=" << last.telemetry.synchronization_count
    << " initial_cost=" << last.initial_cost.total()
    << " final_cost=" << last.final_cost.total()
    << " iterations=" << last.iterations
    << " termination=" << seil::termination_name(last.stop) << '\n';
  return EXIT_SUCCESS;
}

}

int main(int argc, char** argv)
{
  return run(argc, argv);
}
