#include <stdexcept>

#include "axon/ops.h"

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(AXON_HAVE_OPENMP)
#include <omp.h>
#endif

#if defined(AXON_HAVE_CUDA)
#include "axon/backend.h"
namespace axon::cuda {
// Full host round-trip GPU matmul (defined in src/ops/cuda/matmul_cuda.cu).
void cuda_matmul(const float* hA, const float* hB, float* hC, int m, int k, int n);
}  // namespace axon::cuda
#endif

namespace axon::ops {

namespace {

using I = std::int64_t;

// Common signature of a matmul kernel (C comes pre-zeroed; accumulates into C).
using Kernel = void (*)(const float* a, const float* b, float* c, I m, I k, I n);

// Validates 2D inputs, makes them contiguous, and runs `kern` over the buffers.
Tensor run_matmul(const Tensor& a, const Tensor& b, Kernel kern) {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::invalid_argument("matmul: expects 2D tensors");
    }
    const I m = a.shape()[0];
    const I k = a.shape()[1];
    const I n = b.shape()[1];
    if (k != b.shape()[0]) {
        throw std::invalid_argument("matmul: incompatible inner dimensions");
    }
    const Tensor ca = a.contiguous();
    const Tensor cb = b.contiguous();
    Tensor out = Tensor::zeros({m, n});
    kern(ca.data_ptr(), cb.data_ptr(), out.data_ptr(), m, k, n);
    return out;
}

// (1) Naive ijk: dot product per element. Poor access to `b` (by column).
void kernel_naive(const float* a, const float* b, float* c, I m, I k, I n) {
    for (I i = 0; i < m; ++i) {
        for (I j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (I p = 0; p < k; ++p) {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}

// (2) ikj: inner loop scans `b` and `c` sequentially -> vectorizes well (-O3).
void kernel_ikj(const float* a, const float* b, float* c, I m, I k, I n) {
    for (I i = 0; i < m; ++i) {
        for (I p = 0; p < k; ++p) {
            const float aik = a[i * k + p];
            const float* brow = &b[p * n];
            float* crow = &c[i * n];
            for (I j = 0; j < n; ++j) {
                crow[j] += aik * brow[j];
            }
        }
    }
}

// (3) Blocked/tiling: processes blocks that fit in the L1/L2 cache.
void kernel_blocked(const float* a, const float* b, float* c, I m, I k, I n) {
    constexpr I BM = 64, BK = 64, BN = 256;
    for (I ii = 0; ii < m; ii += BM) {
        const I i_end = std::min(ii + BM, m);
        for (I kk = 0; kk < k; kk += BK) {
            const I k_end = std::min(kk + BK, k);
            for (I jj = 0; jj < n; jj += BN) {
                const I j_end = std::min(jj + BN, n);
                for (I i = ii; i < i_end; ++i) {
                    for (I p = kk; p < k_end; ++p) {
                        const float aik = a[i * k + p];
                        const float* brow = &b[p * n];
                        float* crow = &c[i * n];
                        for (I j = jj; j < j_end; ++j) {
                            crow[j] += aik * brow[j];
                        }
                    }
                }
            }
        }
    }
}

// (4) ikj parallelized by rows with OpenMP (each thread handles distinct rows).
void kernel_omp(const float* a, const float* b, float* c, I m, I k, I n) {
#if defined(AXON_HAVE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (I i = 0; i < m; ++i) {
        for (I p = 0; p < k; ++p) {
            const float aik = a[i * k + p];
            const float* brow = &b[p * n];
            float* crow = &c[i * n];
            for (I j = 0; j < n; ++j) {
                crow[j] += aik * brow[j];
            }
        }
    }
}

// (5b) SIMD AVX-512: 16 floats per instruction in the inner loop, + OpenMP.
// Compiled only when the CPU/flags support it (__AVX512F__); otherwise falls back to AVX2/OMP.
void kernel_avx512(const float* a, const float* b, float* c, I m, I k, I n) {
#if defined(__AVX512F__)
#if defined(AXON_HAVE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (I i = 0; i < m; ++i) {
        for (I p = 0; p < k; ++p) {
            const __m512 va = _mm512_set1_ps(a[i * k + p]);
            const float* brow = &b[p * n];
            float* crow = &c[i * n];
            I j = 0;
            for (; j + 16 <= n; j += 16) {
                __m512 vb = _mm512_loadu_ps(brow + j);
                __m512 vc = _mm512_loadu_ps(crow + j);
                vc = _mm512_fmadd_ps(va, vb, vc);
                _mm512_storeu_ps(crow + j, vc);
            }
            for (; j < n; ++j) {
                crow[j] += a[i * k + p] * brow[j];
            }
        }
    }
#else
    kernel_omp(a, b, c, m, k, n);  // no AVX-512 in build: falls back to OpenMP
#endif
}

// (5) Explicit SIMD (AVX2): 8 floats per instruction in the inner loop, + OpenMP.
void kernel_avx2(const float* a, const float* b, float* c, I m, I k, I n) {
#if defined(__AVX2__)
#if defined(AXON_HAVE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (I i = 0; i < m; ++i) {
        for (I p = 0; p < k; ++p) {
            const __m256 va = _mm256_set1_ps(a[i * k + p]);
            const float* brow = &b[p * n];
            float* crow = &c[i * n];
            I j = 0;
            for (; j + 8 <= n; j += 8) {
                __m256 vb = _mm256_loadu_ps(brow + j);
                __m256 vc = _mm256_loadu_ps(crow + j);
                vc = _mm256_fmadd_ps(va, vb, vc);  // c += a*b (fused multiply-add)
                _mm256_storeu_ps(crow + j, vc);
            }
            for (; j < n; ++j) {  // remainder (n not a multiple of 8)
                crow[j] += a[i * k + p] * brow[j];
            }
        }
    }
#else
    kernel_omp(a, b, c, m, k, n);  // no AVX2 in build: falls back to OpenMP
#endif
}

}  // namespace

Tensor matmul(const Tensor& a, const Tensor& b) {
    // For small matrices, the overhead of spawning threads (OpenMP) outweighs the gain.
    // Below a work threshold, use the sequential vectorized version (ikj).
    if (a.ndim() == 2 && b.ndim() == 2) {
        const I work = a.shape()[0] * a.shape()[1] * b.shape()[1];
        if (work < 100000) {
            return run_matmul(a, b, &kernel_ikj);
        }
#if defined(AXON_HAVE_CUDA)
        // Transparent GPU dispatch: big matmuls (fwd + bwd of every Linear/attention)
        // ship to the GPU, so the whole model trains on CUDA with no Python changes.
        if (is_cuda_enabled() && work >= cuda_matmul_threshold()) {
            const I m = a.shape()[0], k = a.shape()[1], n = b.shape()[1];
            if (k == b.shape()[0]) {
                const Tensor ca = a.contiguous();
                const Tensor cb = b.contiguous();
                Tensor out = Tensor::zeros({m, n});
                cuda::cuda_matmul(ca.data_ptr(), cb.data_ptr(), out.data_ptr(),
                                  static_cast<int>(m), static_cast<int>(k), static_cast<int>(n));
                return out;
            }
        }
#endif
    }
#if defined(__AVX512F__)
    return run_matmul(a, b, &kernel_avx512);
#elif defined(__AVX2__)
    return run_matmul(a, b, &kernel_avx2);
#else
    return run_matmul(a, b, &kernel_omp);
#endif
}

namespace detail {
Tensor matmul_naive(const Tensor& a, const Tensor& b) { return run_matmul(a, b, &kernel_naive); }
Tensor matmul_ikj(const Tensor& a, const Tensor& b) { return run_matmul(a, b, &kernel_ikj); }
Tensor matmul_blocked(const Tensor& a, const Tensor& b) { return run_matmul(a, b, &kernel_blocked); }
Tensor matmul_omp(const Tensor& a, const Tensor& b) { return run_matmul(a, b, &kernel_omp); }
Tensor matmul_avx2(const Tensor& a, const Tensor& b) { return run_matmul(a, b, &kernel_avx2); }
Tensor matmul_avx512(const Tensor& a, const Tensor& b) { return run_matmul(a, b, &kernel_avx512); }
}  // namespace detail

}  // namespace axon::ops
