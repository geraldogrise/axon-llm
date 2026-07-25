#include <stdexcept>

#include "axon/ops.h"

namespace axon::ops {

Tensor embedding(const Tensor& weight, const Tensor& idx) {
    if (weight.ndim() != 2) {
        throw std::invalid_argument("embedding: weight must be 2D (num_embeddings, dim)");
    }
    const Tensor cw = weight.contiguous();
    const Tensor ci = idx.contiguous();
    const std::int64_t V = cw.shape()[0];
    const std::int64_t dim = cw.shape()[1];
    const std::int64_t N = ci.numel();

    Tensor out = Tensor::zeros({N, dim});
    const float* pw = cw.data_ptr();
    const float* pi = ci.data_ptr();
    float* po = out.data_ptr();
    for (std::int64_t i = 0; i < N; ++i) {
        const auto t = static_cast<std::int64_t>(pi[i]);
        if (t < 0 || t >= V) {
            throw std::out_of_range("embedding: index out of range");
        }
        for (std::int64_t j = 0; j < dim; ++j) {
            po[i * dim + j] = pw[t * dim + j];
        }
    }
    return out;
}

Tensor embedding_backward(const Tensor& idx, const Tensor& dy, std::int64_t num_embeddings) {
    const Tensor ci = idx.contiguous();
    const Tensor cdy = dy.contiguous();
    const std::int64_t dim = cdy.shape()[1];
    const std::int64_t N = ci.numel();

    Tensor dw = Tensor::zeros({num_embeddings, dim});
    const float* pi = ci.data_ptr();
    const float* pdy = cdy.data_ptr();
    float* pdw = dw.data_ptr();
    // Scatter-add: each used row accumulates the corresponding gradient.
    for (std::int64_t i = 0; i < N; ++i) {
        const auto t = static_cast<std::int64_t>(pi[i]);
        for (std::int64_t j = 0; j < dim; ++j) {
            pdw[t * dim + j] += pdy[i * dim + j];
        }
    }
    return dw;
}

Tensor causal_mask(std::int64_t L) {
    Tensor m = Tensor::zeros({L, L});
    float* p = m.data_ptr();
    for (std::int64_t i = 0; i < L; ++i) {
        for (std::int64_t j = 0; j < L; ++j) {
            p[i * L + j] = (j <= i) ? 0.0f : -1e9f;
        }
    }
    return m;
}

}  // namespace axon::ops
