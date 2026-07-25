#include "axon/optim.h"

#include <cmath>

namespace axon::optim {

void Optimizer::zero_grad() {
    for (Tensor& p : params_) {
        p.zero_grad();
    }
}

// ---------------------------------------------------------------------------
// SGD
// ---------------------------------------------------------------------------
SGD::SGD(std::vector<Tensor> params, float lr, float momentum, float weight_decay)
    : Optimizer(std::move(params)), lr_(lr), momentum_(momentum), weight_decay_(weight_decay) {
    velocity_.resize(params_.size());
    for (std::size_t i = 0; i < params_.size(); ++i) {
        velocity_[i].assign(static_cast<std::size_t>(params_[i].numel()), 0.0f);
    }
}

void SGD::step() {
    for (std::size_t i = 0; i < params_.size(); ++i) {
        Tensor& p = params_[i];
        if (!p.has_grad()) {
            continue;
        }
        float* w = p.data_ptr();
        const float* g = p.grad().data_ptr();
        std::vector<float>& vel = velocity_[i];
        const std::int64_t n = p.numel();
        for (std::int64_t j = 0; j < n; ++j) {
            const float grad = g[j] + weight_decay_ * w[j];  // weight decay (L2)
            vel[static_cast<std::size_t>(j)] =
                momentum_ * vel[static_cast<std::size_t>(j)] + grad;
            w[j] -= lr_ * vel[static_cast<std::size_t>(j)];
        }
    }
}

// ---------------------------------------------------------------------------
// Adam
// ---------------------------------------------------------------------------
Adam::Adam(std::vector<Tensor> params, float lr, float beta1, float beta2, float eps,
           float weight_decay)
    : Optimizer(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps),
      weight_decay_(weight_decay) {
    m_.resize(params_.size());
    v_.resize(params_.size());
    for (std::size_t i = 0; i < params_.size(); ++i) {
        const auto sz = static_cast<std::size_t>(params_[i].numel());
        m_[i].assign(sz, 0.0f);
        v_[i].assign(sz, 0.0f);
    }
}

std::vector<Tensor> Adam::state_dict() const {
    std::vector<Tensor> out;
    out.push_back(Tensor::full({1}, static_cast<float>(t_)));  // contador de passos
    for (std::size_t i = 0; i < params_.size(); ++i) {
        out.push_back(Tensor::from_data(params_[i].shape(), m_[i]));
        out.push_back(Tensor::from_data(params_[i].shape(), v_[i]));
    }
    return out;
}

void Adam::load_state_dict(const std::vector<Tensor>& state) {
    if (state.empty()) return;
    t_ = static_cast<std::int64_t>(state[0].item());
    std::size_t k = 1;
    for (std::size_t i = 0; i < params_.size() && k + 1 < state.size(); ++i) {
        const std::int64_t n = params_[i].numel();
        const float* pm = state[k].data_ptr();
        const float* pv = state[k + 1].data_ptr();
        m_[i].assign(pm, pm + n);
        v_[i].assign(pv, pv + n);
        k += 2;
    }
}

void Adam::step() {
    ++t_;
    const float bc1 = 1.0f - std::pow(beta1_, static_cast<float>(t_));
    const float bc2 = 1.0f - std::pow(beta2_, static_cast<float>(t_));
    for (std::size_t i = 0; i < params_.size(); ++i) {
        Tensor& p = params_[i];
        if (!p.has_grad()) {
            continue;
        }
        float* w = p.data_ptr();
        const float* g = p.grad().data_ptr();
        std::vector<float>& m = m_[i];
        std::vector<float>& v = v_[i];
        const std::int64_t n = p.numel();
        for (std::int64_t j = 0; j < n; ++j) {
            const auto uj = static_cast<std::size_t>(j);
            m[uj] = beta1_ * m[uj] + (1.0f - beta1_) * g[j];
            v[uj] = beta2_ * v[uj] + (1.0f - beta2_) * g[j] * g[j];
            const float m_hat = m[uj] / bc1;
            const float v_hat = v[uj] / bc2;
            w[j] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
            if (weight_decay_ > 0.0f) {
                w[j] -= lr_ * weight_decay_ * w[j];  // decaimento desacoplado (AdamW)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RMSProp
// ---------------------------------------------------------------------------
RMSProp::RMSProp(std::vector<Tensor> params, float lr, float alpha, float eps,
                 float weight_decay)
    : Optimizer(std::move(params)), lr_(lr), alpha_(alpha), eps_(eps),
      weight_decay_(weight_decay) {
    cache_.resize(params_.size());
    for (std::size_t i = 0; i < params_.size(); ++i) {
        cache_[i].assign(static_cast<std::size_t>(params_[i].numel()), 0.0f);
    }
}

void RMSProp::step() {
    for (std::size_t i = 0; i < params_.size(); ++i) {
        Tensor& p = params_[i];
        if (!p.has_grad()) {
            continue;
        }
        float* w = p.data_ptr();
        const float* g = p.grad().data_ptr();
        std::vector<float>& c = cache_[i];
        const std::int64_t n = p.numel();
        for (std::int64_t j = 0; j < n; ++j) {
            const auto uj = static_cast<std::size_t>(j);
            const float grad = g[j] + weight_decay_ * w[j];
            c[uj] = alpha_ * c[uj] + (1.0f - alpha_) * grad * grad;
            w[j] -= lr_ * grad / (std::sqrt(c[uj]) + eps_);
        }
    }
}

}  // namespace axon::optim
