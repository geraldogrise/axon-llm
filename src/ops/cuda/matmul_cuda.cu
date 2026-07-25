// CUDA matmul backend for pyaxon -- runnable (not a stub).
//
// Compiled when configured with -DAXON_ENABLE_CUDA=ON and nvcc is available.
// Provides host-facing entry points that do the full round-trip
// (host -> device -> kernel -> host), so it works from ordinary CPU arrays.
//
// Implementations, fastest first at runtime via `cuda_matmul`:
//   1. cuBLAS  (if built with -DAXON_USE_CUBLAS)  -- production speed
//   2. tiled shared-memory kernel                 -- good, dependency-free
//   3. naive one-thread-per-element kernel         -- reference
//
// On Colab: see notebooks/cuda_colab.ipynb -- it compiles this file with nvcc,
// verifies the result against NumPy and benchmarks the speedup vs CPU.

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#ifdef AXON_USE_CUBLAS
#include <cublas_v2.h>
#endif

namespace axon::cuda {

namespace {
constexpr int kTile = 16;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " +
                                 cudaGetErrorString(e));
    }
}
}  // namespace

// --- kernels ---------------------------------------------------------------

// Naive: one thread per output element. C(m,n) = A(m,k) * B(k,n), row-major.
__global__ void matmul_naive(const float* A, const float* B, float* C,
                             int m, int k, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float acc = 0.0f;
        for (int p = 0; p < k; ++p) acc += A[row * k + p] * B[p * n + col];
        C[row * n + col] = acc;
    }
}

// Tiled: each block cooperatively loads kTile x kTile tiles of A and B into shared
// memory, cutting global-memory traffic ~kTile x. Much faster than naive.
__global__ void matmul_tiled(const float* A, const float* B, float* C,
                             int m, int k, int n) {
    __shared__ float As[kTile][kTile];
    __shared__ float Bs[kTile][kTile];

    const int row = blockIdx.y * kTile + threadIdx.y;
    const int col = blockIdx.x * kTile + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (k + kTile - 1) / kTile; ++t) {
        const int aCol = t * kTile + threadIdx.x;
        const int bRow = t * kTile + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < m && aCol < k) ? A[row * k + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < k && col < n) ? B[bRow * n + col] : 0.0f;
        __syncthreads();
        for (int p = 0; p < kTile; ++p) acc += As[threadIdx.y][p] * Bs[p][threadIdx.x];
        __syncthreads();
    }
    if (row < m && col < n) C[row * n + col] = acc;
}

// --- device-pointer wrappers (dA/dB/dC already on the GPU) ------------------
void matmul_device_naive(const float* dA, const float* dB, float* dC, int m, int k, int n) {
    const dim3 block(kTile, kTile);
    const dim3 grid((n + kTile - 1) / kTile, (m + kTile - 1) / kTile);
    matmul_naive<<<grid, block>>>(dA, dB, dC, m, k, n);
}

void matmul_device_tiled(const float* dA, const float* dB, float* dC, int m, int k, int n) {
    const dim3 block(kTile, kTile);
    const dim3 grid((n + kTile - 1) / kTile, (m + kTile - 1) / kTile);
    matmul_tiled<<<grid, block>>>(dA, dB, dC, m, k, n);
}

// --- host round-trip: works from ordinary CPU arrays -----------------------
// This is what makes the backend usable end-to-end: allocate GPU memory, copy A/B up,
// run the kernel, copy C back, free. C must be preallocated with m*n floats on the host.
void cuda_matmul(const float* hA, const float* hB, float* hC, int m, int k, int n) {
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    const size_t szA = size_t(m) * k * sizeof(float);
    const size_t szB = size_t(k) * n * sizeof(float);
    const size_t szC = size_t(m) * n * sizeof(float);
    check(cudaMalloc(&dA, szA), "cudaMalloc A");
    check(cudaMalloc(&dB, szB), "cudaMalloc B");
    check(cudaMalloc(&dC, szC), "cudaMalloc C");
    check(cudaMemcpy(dA, hA, szA, cudaMemcpyHostToDevice), "memcpy A");
    check(cudaMemcpy(dB, hB, szB, cudaMemcpyHostToDevice), "memcpy B");

#ifdef AXON_USE_CUBLAS
    // cuBLAS is column-major; computing C^T = B^T * A^T by swapping operands lands the
    // row-major result correctly without an explicit transpose. The handle is created
    // once (thread-local) and reused -- creating it per call would cost ~1ms and dominate
    // the training loop's thousands of matmuls.
    static thread_local cublasHandle_t h = [] {
        cublasHandle_t handle;
        cublasCreate(&handle);
        return handle;
    }();
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, dB, n, dA, k, &beta, dC, n);
#else
    matmul_device_tiled(dA, dB, dC, m, k, n);
#endif
    check(cudaGetLastError(), "kernel launch");
    check(cudaDeviceSynchronize(), "sync");
    check(cudaMemcpy(hC, dC, szC, cudaMemcpyDeviceToHost), "memcpy C");
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

// First CUDA device's name, or empty if none is usable.
std::string cuda_device_name() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return {};
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return {};
    return prop.name;
}

}  // namespace axon::cuda
