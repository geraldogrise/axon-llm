#include "axon/backend.h"

#include <atomic>

namespace axon {

// Defined in src/ops/cuda/matmul_cuda.cu (only linked in CUDA builds).
namespace cuda {
std::string cuda_device_name();
}  // namespace cuda

namespace {
// Default GPU matmul crossover. 1024x1024x1024 ~= 1.07e9 MACs; at ~256k the GPU
// round-trip starts paying off on a T4. Kept as an atomic so it can be tuned live.
std::atomic<long long> g_threshold{256LL * 1024};
std::atomic<bool> g_cuda_enabled{true};
}  // namespace

bool cuda_available() {
#if defined(AXON_HAVE_CUDA)
    return true;
#else
    return false;
#endif
}

std::string cuda_device_name() {
#if defined(AXON_HAVE_CUDA)
    return cuda::cuda_device_name();
#else
    return {};
#endif
}

void set_cuda_enabled(bool on) noexcept { g_cuda_enabled.store(on); }
bool is_cuda_enabled() noexcept { return g_cuda_enabled.load(); }

void set_cuda_matmul_threshold(long long work) noexcept { g_threshold.store(work); }
long long cuda_matmul_threshold() noexcept { return g_threshold.load(); }

}  // namespace axon
