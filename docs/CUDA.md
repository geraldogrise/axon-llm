# GPU backends (optional, ready to enable)

pyaxon was designed to accept a GPU backend without rewriting the API. The
**structure is already in place**; GPU compilation is **optional and off by
default** — the normal build stays 100% CPU and requires nothing from NVIDIA/GPU.

All the GPU backends (CUDA, OpenCL, Vulkan) and the cuBLAS/cuDNN libraries
**depend on hardware/SDK** that this machine doesn't have. That's why they were left as
**build options turned off**, with the structure ready:

| CMake flag | What it does |
|------------|-----------|
| `-DAXON_ENABLE_CUDA=ON`   | builds the CUDA kernels (`nvcc`) |
| `-DAXON_ENABLE_OPENCL=ON` | looks for the OpenCL SDK and links |
| `-DAXON_ENABLE_VULKAN=ON` | looks for the Vulkan SDK and links |

Each flag only activates if the toolkit is found; otherwise, it emits a warning and
proceeds with CPU only (the build **never breaks**).

## Current state

| Item | Status |
|------|--------|
| `Device { CPU, CUDA }` enum on the Tensor | ✅ done |
| Build option `AXON_ENABLE_CUDA` (CMake) | ✅ done (OFF by default) |
| Reference CUDA kernel (`src/ops/cuda/matmul_cuda.cu`) | ✅ skeleton |
| Query `axon::cuda_available()` / `ax.cuda_available()` | ✅ done (False without CUDA) |
| Tensor allocation on `Device::CUDA` + host↔device copy | ⬜ to do |
| Per-device dispatch in `ops::matmul`/attention | ⬜ to do |
| cuBLAS/cuDNN integration | ⬜ to do |

## How to enable (when you have an NVIDIA GPU + CUDA Toolkit)

```bash
cmake -S . -B build -G Ninja -DAXON_ENABLE_CUDA=ON
cmake --build build
```

CMake detects the CUDA compiler (`nvcc`); if it exists, it compiles
`src/ops/cuda/matmul_cuda.cu` and defines `AXON_HAVE_CUDA`. If it doesn't exist, it emits a
warning and proceeds with CPU only (the build never breaks because of it).

Check in Python:

```python
import pyaxon as ax
print(ax.cuda_available())   # True if built with CUDA
```

## Next integration steps

1. `Storage` with `cudaMalloc` allocation when `device == CUDA`.
2. `Tensor::to(Device)` — host↔device copy (`cudaMemcpy`).
3. In `ops::matmul` (and in the attention kernels), dispatch to the CUDA version
   when the tensors are on the GPU; otherwise, use the CPU path (SIMD/OpenMP).
4. Swap the naive kernel for **cuBLAS** (performance reference) and compare
   in the benchmarks (CPU vs GPU).

> While there is no CUDA hardware on this machine, the skeleton ensures that adding the
> backend is incremental and non-disruptive.
