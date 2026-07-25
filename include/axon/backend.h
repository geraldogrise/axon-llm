#pragma once

#include <string>

namespace axon {

// Reports whether the library was compiled with CUDA support
// (CMake: -DAXON_ENABLE_CUDA=ON). By default, false (CPU only).
[[nodiscard]] bool cuda_available();

// Name of the first usable CUDA device (e.g. "Tesla T4"), or "" if none/CPU-only.
[[nodiscard]] std::string cuda_device_name();

// Runtime switch for the transparent CUDA matmul dispatch. When the library is
// built with CUDA, large `ops::matmul` calls run on the GPU by default; flip this
// off to force the CPU kernels (useful for A/B benchmarking on Colab). No effect
// on CPU-only builds.
void set_cuda_enabled(bool on) noexcept;
[[nodiscard]] bool is_cuda_enabled() noexcept;

// Minimum work (m*k*n) before matmul is worth shipping to the GPU. Below this the
// host<->device round-trip dominates, so the CPU path stays faster. Tunable so the
// Colab benchmark can sweep the crossover point.
void set_cuda_matmul_threshold(long long work) noexcept;
[[nodiscard]] long long cuda_matmul_threshold() noexcept;

}  // namespace axon
