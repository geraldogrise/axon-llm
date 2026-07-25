#include <cmath>
#include <stdexcept>

#include "axon/ops.h"

namespace axon::ops {

namespace {
void require_2d(const Tensor& t, const char* who) {
    if (t.ndim() != 2) {
        throw std::invalid_argument(std::string(who) + ": expects 2D tensor (batch, features)");
    }
}
}  // namespace

Tensor softmax(const Tensor& a) {
    require_2d(a, "softmax");
    const Tensor ca = a.contiguous();
    const std::int64_t rows = ca.shape()[0];
    const std::int64_t cols = ca.shape()[1];
    Tensor out = Tensor::zeros({rows, cols});
    const float* pa = ca.data_ptr();
    float* po = out.data_ptr();

    for (std::int64_t i = 0; i < rows; ++i) {
        const float* row = &pa[i * cols];
        float* orow = &po[i * cols];
        // Stable: subtracts the row maximum before exponentiating.
        float m = row[0];
        for (std::int64_t j = 1; j < cols; ++j) {
            m = std::max(m, row[j]);
        }
        float sum = 0.0f;
        for (std::int64_t j = 0; j < cols; ++j) {
            orow[j] = std::exp(row[j] - m);
            sum += orow[j];
        }
        for (std::int64_t j = 0; j < cols; ++j) {
            orow[j] /= sum;
        }
    }
    return out;
}

Tensor softmax_grad(const Tensor& y, const Tensor& g) {
    // For each row: dx = y * (g - sum(g * y)).
    require_2d(y, "softmax_grad");
    const Tensor cy = y.contiguous();
    const Tensor cg = g.contiguous();
    const std::int64_t rows = cy.shape()[0];
    const std::int64_t cols = cy.shape()[1];
    Tensor out = Tensor::zeros({rows, cols});
    const float* py = cy.data_ptr();
    const float* pg = cg.data_ptr();
    float* po = out.data_ptr();

    for (std::int64_t i = 0; i < rows; ++i) {
        const float* yrow = &py[i * cols];
        const float* grow = &pg[i * cols];
        float* orow = &po[i * cols];
        float dot = 0.0f;
        for (std::int64_t j = 0; j < cols; ++j) {
            dot += grow[j] * yrow[j];
        }
        for (std::int64_t j = 0; j < cols; ++j) {
            orow[j] = yrow[j] * (grow[j] - dot);
        }
    }
    return out;
}

Tensor log_softmax(const Tensor& a) {
    require_2d(a, "log_softmax");
    const Tensor ca = a.contiguous();
    const std::int64_t rows = ca.shape()[0];
    const std::int64_t cols = ca.shape()[1];
    Tensor out = Tensor::zeros({rows, cols});
    const float* pa = ca.data_ptr();
    float* po = out.data_ptr();
    for (std::int64_t i = 0; i < rows; ++i) {
        const float* row = &pa[i * cols];
        float m = row[0];
        for (std::int64_t j = 1; j < cols; ++j) m = std::max(m, row[j]);
        float sum = 0.0f;
        for (std::int64_t j = 0; j < cols; ++j) sum += std::exp(row[j] - m);
        const float logsumexp = m + std::log(sum);
        for (std::int64_t j = 0; j < cols; ++j) po[i * cols + j] = row[j] - logsumexp;
    }
    return out;
}

Tensor log_softmax_grad(const Tensor& y, const Tensor& g) {
    // y = log_softmax(x); dx = g - softmax(x) * sum(g). softmax = exp(y).
    require_2d(y, "log_softmax_grad");
    const Tensor cy = y.contiguous();
    const Tensor cg = g.contiguous();
    const std::int64_t rows = cy.shape()[0];
    const std::int64_t cols = cy.shape()[1];
    Tensor out = Tensor::zeros({rows, cols});
    const float* py = cy.data_ptr();
    const float* pg = cg.data_ptr();
    float* po = out.data_ptr();
    for (std::int64_t i = 0; i < rows; ++i) {
        float gsum = 0.0f;
        for (std::int64_t j = 0; j < cols; ++j) gsum += pg[i * cols + j];
        for (std::int64_t j = 0; j < cols; ++j) {
            const float sm = std::exp(py[i * cols + j]);
            po[i * cols + j] = pg[i * cols + j] - sm * gsum;
        }
    }
    return out;
}

Tensor cross_entropy(const Tensor& logits, const Tensor& targets) {
    require_2d(logits, "cross_entropy");
    const Tensor cl = logits.contiguous();
    const Tensor ct = targets.contiguous();
    const std::int64_t rows = cl.shape()[0];
    const std::int64_t cols = cl.shape()[1];
    if (ct.numel() != rows) {
        throw std::invalid_argument("cross_entropy: targets must have N elements (1 per row)");
    }
    const float* pl = cl.data_ptr();
    const float* pt = ct.data_ptr();

    float loss = 0.0f;
    for (std::int64_t i = 0; i < rows; ++i) {
        const float* row = &pl[i * cols];
        // stable log-sum-exp.
        float m = row[0];
        for (std::int64_t j = 1; j < cols; ++j) {
            m = std::max(m, row[j]);
        }
        float sum = 0.0f;
        for (std::int64_t j = 0; j < cols; ++j) {
            sum += std::exp(row[j] - m);
        }
        const float logsumexp = m + std::log(sum);
        const auto t = static_cast<std::int64_t>(pt[i]);
        if (t < 0 || t >= cols) {
            throw std::out_of_range("cross_entropy: class index out of range");
        }
        loss += logsumexp - row[t];  // -log(softmax[t])
    }
    return Tensor::full({1}, loss / static_cast<float>(rows));
}

Tensor cross_entropy_grad(const Tensor& logits, const Tensor& targets) {
    // dL/dlogits = (softmax - onehot(target)) / N.
    require_2d(logits, "cross_entropy_grad");
    Tensor probs = softmax(logits);
    const Tensor ct = targets.contiguous();
    const std::int64_t rows = probs.shape()[0];
    const std::int64_t cols = probs.shape()[1];
    const float* pt = ct.data_ptr();
    float* pp = probs.data_ptr();
    const float inv_n = 1.0f / static_cast<float>(rows);
    for (std::int64_t i = 0; i < rows; ++i) {
        const auto t = static_cast<std::int64_t>(pt[i]);
        pp[i * cols + t] -= 1.0f;
        for (std::int64_t j = 0; j < cols; ++j) {
            pp[i * cols + j] *= inv_n;
        }
    }
    return probs;
}

// ----- Binary cross-entropy with logits -----
Tensor bce_with_logits(const Tensor& pred, const Tensor& target) {
    const Tensor cp = pred.contiguous();
    const Tensor cy = target.contiguous();
    if (cp.numel() != cy.numel()) {
        throw std::invalid_argument("bce_with_logits: pred and target must have same size");
    }
    const std::int64_t n = cp.numel();
    const float* x = cp.data_ptr();
    const float* y = cy.data_ptr();
    float loss = 0.0f;
    for (std::int64_t i = 0; i < n; ++i) {
        // stable form: max(x,0) - x*y + log(1 + exp(-|x|))
        loss += std::max(x[i], 0.0f) - x[i] * y[i] + std::log1p(std::exp(-std::abs(x[i])));
    }
    return Tensor::full({1}, loss / static_cast<float>(n));
}

Tensor bce_with_logits_grad(const Tensor& pred, const Tensor& target) {
    const Tensor cp = pred.contiguous();
    const Tensor cy = target.contiguous();
    const std::int64_t n = cp.numel();
    const float* x = cp.data_ptr();
    const float* y = cy.data_ptr();
    Tensor out = Tensor::zeros(pred.shape());
    float* g = out.data_ptr();
    const float inv_n = 1.0f / static_cast<float>(n);
    for (std::int64_t i = 0; i < n; ++i) {
        const float s = 1.0f / (1.0f + std::exp(-x[i]));   // sigmoid
        g[i] = (s - y[i]) * inv_n;
    }
    return out;
}

// ----- Huber loss (smooth L1) -----
Tensor huber(const Tensor& pred, const Tensor& target, float delta) {
    const Tensor cp = pred.contiguous();
    const Tensor cy = target.contiguous();
    if (cp.numel() != cy.numel()) {
        throw std::invalid_argument("huber: pred and target must have same size");
    }
    const std::int64_t n = cp.numel();
    const float* p = cp.data_ptr();
    const float* t = cy.data_ptr();
    float loss = 0.0f;
    for (std::int64_t i = 0; i < n; ++i) {
        const float d = p[i] - t[i];
        const float a = std::abs(d);
        loss += (a <= delta) ? 0.5f * d * d : delta * (a - 0.5f * delta);
    }
    return Tensor::full({1}, loss / static_cast<float>(n));
}

Tensor huber_grad(const Tensor& pred, const Tensor& target, float delta) {
    const Tensor cp = pred.contiguous();
    const Tensor cy = target.contiguous();
    const std::int64_t n = cp.numel();
    const float* p = cp.data_ptr();
    const float* t = cy.data_ptr();
    Tensor out = Tensor::zeros(pred.shape());
    float* g = out.data_ptr();
    const float inv_n = 1.0f / static_cast<float>(n);
    for (std::int64_t i = 0; i < n; ++i) {
        const float d = p[i] - t[i];
        const float gd = (std::abs(d) <= delta) ? d : delta * ((d > 0) ? 1.0f : -1.0f);
        g[i] = gd * inv_n;
    }
    return out;
}

}  // namespace axon::ops
