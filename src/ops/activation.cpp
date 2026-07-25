#include <cmath>
#include <stdexcept>

#include "axon/ops.h"

#if defined(AXON_HAVE_OPENMP)
#include <omp.h>
#endif

namespace axon::ops {

namespace {
constexpr std::int64_t kParallelThreshold = 50000;

// Applies a scalar function element-wise over a contiguous tensor.
template <typename F>
Tensor unary(const Tensor& a, F f) {
    const Tensor ca = a.contiguous();
    Tensor out = Tensor::zeros(ca.shape());
    const float* p = ca.data_ptr();
    float* po = out.data_ptr();
    const std::int64_t n = out.numel();
#if defined(AXON_HAVE_OPENMP)
#pragma omp parallel for schedule(static) if (n > kParallelThreshold)
#endif
    for (std::int64_t i = 0; i < n; ++i) {
        po[i] = f(p[i]);
    }
    return out;
}

// Combines two tensors of the same shape element-wise.
template <typename F>
Tensor binary_same(const Tensor& a, const Tensor& b, F f) {
    const Tensor ca = a.contiguous();
    const Tensor cb = b.contiguous();
    Tensor out = Tensor::zeros(ca.shape());
    const float* pa = ca.data_ptr();
    const float* pb = cb.data_ptr();
    float* po = out.data_ptr();
    const std::int64_t n = out.numel();
#if defined(AXON_HAVE_OPENMP)
#pragma omp parallel for schedule(static) if (n > kParallelThreshold)
#endif
    for (std::int64_t i = 0; i < n; ++i) {
        po[i] = f(pa[i], pb[i]);
    }
    return out;
}

float sigmoid_scalar(float x) {
    // Stable form: avoids exp overflow for very negative x.
    if (x >= 0.0f) {
        return 1.0f / (1.0f + std::exp(-x));
    }
    const float e = std::exp(x);
    return e / (1.0f + e);
}
}  // namespace

Tensor relu(const Tensor& a) {
    return unary(a, [](float x) { return x > 0.0f ? x : 0.0f; });
}

Tensor relu_mask(const Tensor& a) {
    return unary(a, [](float x) { return x > 0.0f ? 1.0f : 0.0f; });
}

Tensor sigmoid(const Tensor& a) {
    return unary(a, sigmoid_scalar);
}

Tensor sigmoid_grad(const Tensor& y, const Tensor& g) {
    return binary_same(y, g, [](float yi, float gi) { return gi * yi * (1.0f - yi); });
}

Tensor tanh(const Tensor& a) {
    return unary(a, [](float x) { return std::tanh(x); });
}

Tensor tanh_grad(const Tensor& y, const Tensor& g) {
    return binary_same(y, g, [](float yi, float gi) { return gi * (1.0f - yi * yi); });
}

Tensor gelu(const Tensor& a) {
    // Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2))).
    constexpr float inv_sqrt2 = 0.70710678f;
    return unary(a, [](float x) { return 0.5f * x * (1.0f + std::erf(x * inv_sqrt2)); });
}

namespace {
// Applies (x + bias) followed by an activation, in a single pass (fusion).
template <typename Act>
Tensor bias_act(const Tensor& x, const Tensor& bias, Act act) {
    const Tensor cx = x.contiguous();
    const Tensor cb = bias.contiguous();
    if (cx.ndim() != 2 || cb.numel() != cx.shape()[1]) {
        throw std::invalid_argument("bias_act: x must be (N,D) and bias (D,)");
    }
    const std::int64_t rows = cx.shape()[0];
    const std::int64_t cols = cx.shape()[1];
    Tensor out = Tensor::zeros({rows, cols});
    const float* px = cx.data_ptr();
    const float* pb = cb.data_ptr();
    float* po = out.data_ptr();
#if defined(AXON_HAVE_OPENMP)
#pragma omp parallel for schedule(static) if (rows * cols > kParallelThreshold)
#endif
    for (std::int64_t i = 0; i < rows; ++i) {
        for (std::int64_t j = 0; j < cols; ++j) {
            po[i * cols + j] = act(px[i * cols + j] + pb[j]);
        }
    }
    return out;
}
}  // namespace

Tensor bias_relu(const Tensor& x, const Tensor& bias) {
    return bias_act(x, bias, [](float v) { return v > 0.0f ? v : 0.0f; });
}

Tensor bias_gelu(const Tensor& x, const Tensor& bias) {
    constexpr float inv_sqrt2 = 0.70710678f;
    return bias_act(x, bias,
                    [](float v) { return 0.5f * v * (1.0f + std::erf(v * inv_sqrt2)); });
}

Tensor gelu_grad(const Tensor& x, const Tensor& g) {
    // d/dx GELU = 0.5(1 + erf(x/sqrt2)) + x * (1/sqrt(2pi)) * exp(-x^2/2).
    constexpr float inv_sqrt2 = 0.70710678f;
    constexpr float inv_sqrt2pi = 0.39894228f;
    return binary_same(x, g, [](float xi, float gi) {
        const float cdf = 0.5f * (1.0f + std::erf(xi * inv_sqrt2));
        const float pdf = inv_sqrt2pi * std::exp(-0.5f * xi * xi);
        return gi * (cdf + xi * pdf);
    });
}

}  // namespace axon::ops
