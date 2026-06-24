#include "../src/cuda_backend.hpp"

#include <seil/problem.hpp>

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
  seil::problem graph;
  graph.nodes.push_back({1, {}, true});
  graph.nodes.push_back({2, {1.0, 0.0, 0.0, 0.0}, false});
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
#endif
  int32_t count = -1;
  const int status = seil_cuda_device_count(&count);
  require(status == 0 || count == 0, "CUDA device query status");
}
