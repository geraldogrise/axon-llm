#include "axon/functional.h"

#include <cmath>
#include <memory>
#include <vector>

#include "axon/autograd.h"
#include "axon/ops.h"

namespace axon {

namespace {

// Attaches `node` as `value`'s grad_fn if any input requires a gradient AND
// autograd is enabled (inside no_grad, it does not build a graph).
Tensor with_grad(Tensor value, const std::shared_ptr<Node>& node, bool needs_grad) {
    if (needs_grad && is_grad_enabled()) {
        value.set_grad_fn(node);
    }
    return value;
}

bool any_requires_grad(std::initializer_list<const Tensor*> ts) {
    for (const Tensor* t : ts) {
        if (t->requires_grad()) {
            return true;
        }
    }
    return false;
}

// ----- Graph nodes -----
struct MatmulNode : Node {
    std::vector<Tensor> backward(const Tensor& g) override {
        const Tensor& a = inputs[0];
        const Tensor& b = inputs[1];
        Tensor ga = ops::matmul(g, b.transpose(0, 1));  // (m,n) @ (n,k) -> (m,k)
        Tensor gb = ops::matmul(a.transpose(0, 1), g);  // (k,m) @ (m,n) -> (k,n)
        return {ga, gb};
    }
    const char* name() const override { return "Matmul"; }
};

struct AddNode : Node {
    std::vector<Tensor> backward(const Tensor& g) override {
        return {ops::sum_to(g, inputs[0].shape()), ops::sum_to(g, inputs[1].shape())};
    }
    const char* name() const override { return "Add"; }
};

struct SubNode : Node {
    std::vector<Tensor> backward(const Tensor& g) override {
        return {ops::sum_to(g, inputs[0].shape()),
                ops::sum_to(ops::neg(g), inputs[1].shape())};
    }
    const char* name() const override { return "Sub"; }
};

struct MulNode : Node {
    std::vector<Tensor> backward(const Tensor& g) override {
        Tensor ga = ops::sum_to(ops::mul(g, inputs[1]), inputs[0].shape());
        Tensor gb = ops::sum_to(ops::mul(g, inputs[0]), inputs[1].shape());
        return {ga, gb};
    }
    const char* name() const override { return "Mul"; }
};

struct ReluNode : Node {
    std::vector<Tensor> backward(const Tensor& g) override {
        return {ops::mul(g, ops::relu_mask(inputs[0]))};
    }
    const char* name() const override { return "Relu"; }
};

// Activations that need the output (y) in backward store it in `saved`.
struct SigmoidNode : Node {
    Tensor saved;  // y = sigmoid(x)
    std::vector<Tensor> backward(const Tensor& g) override {
        return {ops::sigmoid_grad(saved, g)};
    }
    const char* name() const override { return "Sigmoid"; }
};

struct TanhNode : Node {
    Tensor saved;  // y = tanh(x)
    std::vector<Tensor> backward(const Tensor& g) override {
        return {ops::tanh_grad(saved, g)};
    }
    const char* name() const override { return "Tanh"; }
};

struct GeluNode : Node {
    std::vector<Tensor> backward(const Tensor& g) override {
        return {ops::gelu_grad(inputs[0], g)};  // uses x (the input)
    }
    const char* name() const override { return "Gelu"; }
};

struct SoftmaxNode : Node {
    Tensor saved;  // y = softmax(x)
    std::vector<Tensor> backward(const Tensor& g) override {
        return {ops::softmax_grad(saved, g)};
    }
    const char* name() const override { return "Softmax"; }
};

struct LogSoftmaxNode : Node {
    Tensor saved;  // y = log_softmax(x)
    std::vector<Tensor> backward(const Tensor& g) override {
        return {ops::log_softmax_grad(saved, g)};
    }
    const char* name() const override { return "LogSoftmax"; }
};

struct CrossEntropyNode : Node {
    // inputs = {logits, targets}. Backward: dlogits = (softmax - onehot)/N * g.
    std::vector<Tensor> backward(const Tensor& g) override {
        Tensor dlogits = ops::mul_scalar(ops::cross_entropy_grad(inputs[0], inputs[1]), g.item());
        return {dlogits, Tensor::zeros(inputs[1].shape())};
    }
    const char* name() const override { return "CrossEntropy"; }
};

struct BCENode : Node {
    // inputs = {pred(logits), target}. Backward: dpred = (sigmoid(x)-y)/N * g.
    std::vector<Tensor> backward(const Tensor& g) override {
        Tensor dpred = ops::mul_scalar(ops::bce_with_logits_grad(inputs[0], inputs[1]), g.item());
        return {dpred, Tensor::zeros(inputs[1].shape())};
    }
    const char* name() const override { return "BCE"; }
};

struct HuberNode : Node {
    float delta = 1.0f;
    std::vector<Tensor> backward(const Tensor& g) override {
        Tensor dpred = ops::mul_scalar(ops::huber_grad(inputs[0], inputs[1], delta), g.item());
        return {dpred, Tensor::zeros(inputs[1].shape())};
    }
    const char* name() const override { return "Huber"; }
};

struct Conv2dNode : Node {
    // inputs = {x, weight, bias}. Backward returns {dx, dweight, dbias}.
    int stride = 1;
    int pad = 0;
    std::vector<Tensor> backward(const Tensor& g) override {
        return ops::conv2d_backward(inputs[0], inputs[1], g, stride, pad);
    }
    const char* name() const override { return "Conv2d"; }
};

struct TransposeNode : Node {
    std::int64_t d0 = 0, d1 = 1;
    std::vector<Tensor> backward(const Tensor& g) override {
        // The transpose is its own inverse: returns the transposed grad back.
        return {g.transpose(d0, d1).contiguous()};
    }
    const char* name() const override { return "Transpose"; }
};

struct ScaleNode : Node {
    float s = 1.0f;
    std::vector<Tensor> backward(const Tensor& g) override { return {ops::mul_scalar(g, s)}; }
    const char* name() const override { return "Scale"; }
};

struct LayerNormNode : Node {
    // inputs = {x, gamma, beta}. Stores xhat/rstd and gamma for the backward.
    ops::LayerNormCache cache;
    std::vector<Tensor> backward(const Tensor& g) override {
        Tensor dx, dgamma, dbeta;
        ops::layernorm_backward(g, cache.xhat, cache.rstd, inputs[1], dx, dgamma, dbeta);
        return {dx, dgamma, dbeta};
    }
    const char* name() const override { return "LayerNorm"; }
};

struct EmbeddingNode : Node {
    // inputs = {weight, idx}. Backward scatters the grad back into the used rows.
    std::int64_t num_embeddings = 0;
    std::vector<Tensor> backward(const Tensor& g) override {
        Tensor dw = ops::embedding_backward(inputs[1], g, num_embeddings);
        return {dw, Tensor::zeros(inputs[1].shape())};
    }
    const char* name() const override { return "Embedding"; }
};

struct NarrowColsNode : Node {
    std::int64_t start = 0, full_cols = 0;
    std::vector<Tensor> backward(const Tensor& g) override {
        return {ops::slice_cols_backward(g, start, full_cols)};
    }
    const char* name() const override { return "NarrowCols"; }
};

struct ConcatColsNode : Node {
    std::vector<std::int64_t> widths;  // width of each part, in order
    std::vector<Tensor> backward(const Tensor& g) override {
        std::vector<Tensor> grads;
        std::int64_t off = 0;
        for (std::int64_t w : widths) {
            grads.push_back(ops::slice_cols(g, off, w));
            off += w;
        }
        return grads;
    }
    const char* name() const override { return "ConcatCols"; }
};

struct NarrowRowsNode : Node {
    std::int64_t start = 0, full_rows = 0;
    std::vector<Tensor> backward(const Tensor& g) override {
        return {ops::slice_rows_backward(g, start, full_rows)};
    }
    const char* name() const override { return "NarrowRows"; }
};

struct ConcatRowsNode : Node {
    std::vector<std::int64_t> heights;  // number of rows of each part
    std::vector<Tensor> backward(const Tensor& g) override {
        std::vector<Tensor> grads;
        std::int64_t off = 0;
        for (std::int64_t h : heights) {
            grads.push_back(ops::slice_rows(g, off, h));
            off += h;
        }
        return grads;
    }
    const char* name() const override { return "ConcatRows"; }
};

struct SumNode : Node {
    std::vector<Tensor> backward(const Tensor& g) override {
        // g is a scalar {1}; spread it across the entire input shape.
        return {Tensor::full(inputs[0].shape(), g.item())};
    }
    const char* name() const override { return "Sum"; }
};

struct MeanNode : Node {
    std::int64_t n = 1;
    std::vector<Tensor> backward(const Tensor& g) override {
        return {Tensor::full(inputs[0].shape(), g.item() / static_cast<float>(n))};
    }
    const char* name() const override { return "Mean"; }
};

}  // namespace

namespace fn {

Tensor matmul(const Tensor& a, const Tensor& b) {
    Tensor out = ops::matmul(a, b);
    const bool need = any_requires_grad({&a, &b});
    auto node = std::make_shared<MatmulNode>();
    node->inputs = {a, b};
    return with_grad(std::move(out), node, need);
}

Tensor add(const Tensor& a, const Tensor& b) {
    Tensor out = ops::add(a, b);
    const bool need = any_requires_grad({&a, &b});
    auto node = std::make_shared<AddNode>();
    node->inputs = {a, b};
    return with_grad(std::move(out), node, need);
}

Tensor sub(const Tensor& a, const Tensor& b) {
    Tensor out = ops::sub(a, b);
    const bool need = any_requires_grad({&a, &b});
    auto node = std::make_shared<SubNode>();
    node->inputs = {a, b};
    return with_grad(std::move(out), node, need);
}

Tensor mul(const Tensor& a, const Tensor& b) {
    Tensor out = ops::mul(a, b);
    const bool need = any_requires_grad({&a, &b});
    auto node = std::make_shared<MulNode>();
    node->inputs = {a, b};
    return with_grad(std::move(out), node, need);
}

Tensor relu(const Tensor& a) {
    Tensor out = ops::relu(a);
    const bool need = a.requires_grad();
    auto node = std::make_shared<ReluNode>();
    node->inputs = {a};
    return with_grad(std::move(out), node, need);
}

Tensor sigmoid(const Tensor& a) {
    Tensor out = ops::sigmoid(a);
    const bool need = a.requires_grad();
    auto node = std::make_shared<SigmoidNode>();
    node->inputs = {a};
    node->saved = out;  // store y for the backward
    return with_grad(std::move(out), node, need);
}

Tensor tanh(const Tensor& a) {
    Tensor out = ops::tanh(a);
    const bool need = a.requires_grad();
    auto node = std::make_shared<TanhNode>();
    node->inputs = {a};
    node->saved = out;
    return with_grad(std::move(out), node, need);
}

Tensor gelu(const Tensor& a) {
    Tensor out = ops::gelu(a);
    const bool need = a.requires_grad();
    auto node = std::make_shared<GeluNode>();
    node->inputs = {a};
    return with_grad(std::move(out), node, need);
}

Tensor softmax(const Tensor& a) {
    Tensor out = ops::softmax(a);
    const bool need = a.requires_grad();
    auto node = std::make_shared<SoftmaxNode>();
    node->inputs = {a};
    node->saved = out;
    return with_grad(std::move(out), node, need);
}

Tensor log_softmax(const Tensor& a) {
    Tensor out = ops::log_softmax(a);
    const bool need = a.requires_grad();
    auto node = std::make_shared<LogSoftmaxNode>();
    node->inputs = {a};
    node->saved = out;
    return with_grad(std::move(out), node, need);
}

Tensor sum(const Tensor& a) {
    Tensor out = ops::sum_all(a);
    const bool need = a.requires_grad();
    auto node = std::make_shared<SumNode>();
    node->inputs = {a};
    return with_grad(std::move(out), node, need);
}

Tensor mean(const Tensor& a) {
    Tensor out = ops::mean_all(a);
    const bool need = a.requires_grad();
    auto node = std::make_shared<MeanNode>();
    node->n = a.numel();
    node->inputs = {a};
    return with_grad(std::move(out), node, need);
}

Tensor mse_loss(const Tensor& pred, const Tensor& target) {
    Tensor d = sub(pred, target);
    return mean(mul(d, d));
}

Tensor cross_entropy(const Tensor& logits, const Tensor& targets) {
    Tensor out = ops::cross_entropy(logits, targets);
    const bool need = logits.requires_grad();
    auto node = std::make_shared<CrossEntropyNode>();
    node->inputs = {logits, targets};
    return with_grad(std::move(out), node, need);
}

Tensor bce_loss(const Tensor& pred, const Tensor& target) {
    Tensor out = ops::bce_with_logits(pred, target);
    const bool need = pred.requires_grad();
    auto node = std::make_shared<BCENode>();
    node->inputs = {pred, target};
    return with_grad(std::move(out), node, need);
}

Tensor huber_loss(const Tensor& pred, const Tensor& target, float delta) {
    Tensor out = ops::huber(pred, target, delta);
    const bool need = pred.requires_grad();
    auto node = std::make_shared<HuberNode>();
    node->delta = delta;
    node->inputs = {pred, target};
    return with_grad(std::move(out), node, need);
}

Tensor conv2d(const Tensor& x, const Tensor& weight, const Tensor& bias, int stride, int pad) {
    Tensor out = ops::conv2d(x, weight, bias, stride, pad);
    const bool need = x.requires_grad() || weight.requires_grad() || bias.requires_grad();
    auto node = std::make_shared<Conv2dNode>();
    node->stride = stride;
    node->pad = pad;
    node->inputs = {x, weight, bias};
    return with_grad(std::move(out), node, need);
}

Tensor transpose(const Tensor& a, std::int64_t dim0, std::int64_t dim1) {
    Tensor out = a.transpose(dim0, dim1).contiguous();
    const bool need = a.requires_grad();
    auto node = std::make_shared<TransposeNode>();
    node->d0 = dim0;
    node->d1 = dim1;
    node->inputs = {a};
    return with_grad(std::move(out), node, need);
}

Tensor scale(const Tensor& a, float s) {
    Tensor out = ops::mul_scalar(a, s);
    const bool need = a.requires_grad();
    auto node = std::make_shared<ScaleNode>();
    node->s = s;
    node->inputs = {a};
    return with_grad(std::move(out), node, need);
}

Tensor layernorm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps) {
    auto node = std::make_shared<LayerNormNode>();
    Tensor out = ops::layernorm(x, gamma, beta, eps, &node->cache);
    const bool need = x.requires_grad() || gamma.requires_grad() || beta.requires_grad();
    node->inputs = {x, gamma, beta};
    return with_grad(std::move(out), node, need);
}

Tensor embedding(const Tensor& weight, const Tensor& idx) {
    Tensor out = ops::embedding(weight, idx);
    const bool need = weight.requires_grad();
    auto node = std::make_shared<EmbeddingNode>();
    node->num_embeddings = weight.shape()[0];
    node->inputs = {weight, idx};
    return with_grad(std::move(out), node, need);
}

Tensor scaled_dot_product_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                    bool causal) {
    const std::int64_t d = q.shape()[1];
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
    // scores = (Q Kᵀ) / sqrt(d)
    Tensor scores = scale(matmul(q, transpose(k, 0, 1)), inv_sqrt_d);
    if (causal) {
        // add causal mask (constant): future positions become -inf before the softmax.
        scores = add(scores, ops::causal_mask(scores.shape()[0]));
    }
    Tensor attn = softmax(scores);  // (Lq, Lk)
    return matmul(attn, v);         // (Lq, dv)
}

Tensor narrow_cols(const Tensor& a, std::int64_t start, std::int64_t len) {
    Tensor out = ops::slice_cols(a, start, len);
    const bool need = a.requires_grad();
    auto node = std::make_shared<NarrowColsNode>();
    node->start = start;
    node->full_cols = a.shape()[1];
    node->inputs = {a};
    return with_grad(std::move(out), node, need);
}

Tensor cat_cols(const std::vector<Tensor>& parts) {
    Tensor out = ops::concat_cols(parts);
    bool need = false;
    auto node = std::make_shared<ConcatColsNode>();
    for (const Tensor& p : parts) {
        need = need || p.requires_grad();
        node->widths.push_back(p.shape()[1]);
    }
    node->inputs = parts;
    return with_grad(std::move(out), node, need);
}

Tensor narrow_rows(const Tensor& a, std::int64_t start, std::int64_t len) {
    Tensor out = ops::slice_rows(a, start, len);
    const bool need = a.requires_grad();
    auto node = std::make_shared<NarrowRowsNode>();
    node->start = start;
    node->full_rows = a.shape()[0];
    node->inputs = {a};
    return with_grad(std::move(out), node, need);
}

Tensor cat_rows(const std::vector<Tensor>& parts) {
    Tensor out = ops::concat_rows(parts);
    bool need = false;
    auto node = std::make_shared<ConcatRowsNode>();
    for (const Tensor& p : parts) {
        need = need || p.requires_grad();
        node->heights.push_back(p.shape()[0]);
    }
    node->inputs = parts;
    return with_grad(std::move(out), node, need);
}

}  // namespace fn

// ----- Operators -----
Tensor operator+(const Tensor& a, const Tensor& b) { return fn::add(a, b); }
Tensor operator-(const Tensor& a, const Tensor& b) { return fn::sub(a, b); }
Tensor operator*(const Tensor& a, const Tensor& b) { return fn::mul(a, b); }

}  // namespace axon
