#include <cmath>
#include <stdexcept>

#include "axon/ops.h"

namespace axon::ops {

namespace {
void require_2d(const Tensor& t, const char* who) {
    if (t.ndim() != 2) {
        throw std::invalid_argument(std::string(who) + ": expects 2D tensor (N, D)");
    }
}
}  // namespace

Tensor layernorm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps,
                 LayerNormCache* cache) {
    require_2d(x, "layernorm");
    const Tensor cx = x.contiguous();
    const Tensor cg = gamma.contiguous();
    const Tensor cb = beta.contiguous();
    const std::int64_t N = cx.shape()[0];
    const std::int64_t D = cx.shape()[1];
    if (cg.numel() != D || cb.numel() != D) {
        throw std::invalid_argument("layernorm: gamma/beta must have size D");
    }

    Tensor out = Tensor::zeros({N, D});
    Tensor xhat = Tensor::zeros({N, D});
    Tensor rstd = Tensor::zeros({N, 1});
    const float* px = cx.data_ptr();
    const float* pg = cg.data_ptr();
    const float* pb = cb.data_ptr();
    float* po = out.data_ptr();
    float* pxh = xhat.data_ptr();
    float* pr = rstd.data_ptr();

    for (std::int64_t i = 0; i < N; ++i) {
        const float* row = &px[i * D];
        // mean and variance of the row.
        float mean = 0.0f;
        for (std::int64_t j = 0; j < D; ++j) {
            mean += row[j];
        }
        mean /= static_cast<float>(D);
        float var = 0.0f;
        for (std::int64_t j = 0; j < D; ++j) {
            const float d = row[j] - mean;
            var += d * d;
        }
        var /= static_cast<float>(D);
        const float r = 1.0f / std::sqrt(var + eps);
        pr[i] = r;
        for (std::int64_t j = 0; j < D; ++j) {
            const float xh = (row[j] - mean) * r;
            pxh[i * D + j] = xh;
            po[i * D + j] = xh * pg[j] + pb[j];
        }
    }

    if (cache != nullptr) {
        cache->xhat = xhat;
        cache->rstd = rstd;
    }
    return out;
}

void layernorm_backward(const Tensor& dy, const Tensor& xhat, const Tensor& rstd,
                        const Tensor& gamma, Tensor& dx, Tensor& dgamma, Tensor& dbeta) {
    const Tensor cdy = dy.contiguous();
    const Tensor cxh = xhat.contiguous();
    const Tensor cg = gamma.contiguous();
    const std::int64_t N = cdy.shape()[0];
    const std::int64_t D = cdy.shape()[1];

    dx = Tensor::zeros({N, D});
    dgamma = Tensor::zeros({D});
    dbeta = Tensor::zeros({D});
    const float* pdy = cdy.data_ptr();
    const float* pxh = cxh.data_ptr();
    const float* pg = cg.data_ptr();
    const float* pr = rstd.contiguous().data_ptr();
    float* pdx = dx.data_ptr();
    float* pdg = dgamma.data_ptr();
    float* pdb = dbeta.data_ptr();

    for (std::int64_t i = 0; i < N; ++i) {
        const float r = pr[i];
        // dxhat = dy * gamma; accumulate dgamma/dbeta.
        float sum_dxhat = 0.0f;
        float sum_dxhat_xhat = 0.0f;
        for (std::int64_t j = 0; j < D; ++j) {
            const float dyij = pdy[i * D + j];
            const float xh = pxh[i * D + j];
            const float dxhat = dyij * pg[j];
            sum_dxhat += dxhat;
            sum_dxhat_xhat += dxhat * xh;
            pdg[j] += dyij * xh;  // dgamma
            pdb[j] += dyij;       // dbeta
        }
        const float invD = 1.0f / static_cast<float>(D);
        for (std::int64_t j = 0; j < D; ++j) {
            const float dxhat = pdy[i * D + j] * pg[j];
            const float xh = pxh[i * D + j];
            // dx = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat*xhat))
            pdx[i * D + j] = r * (dxhat - invD * sum_dxhat - xh * invD * sum_dxhat_xhat);
        }
    }
}

}  // namespace axon::ops
