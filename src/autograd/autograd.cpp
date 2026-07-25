#include "axon/autograd.h"

#include <functional>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "axon/ops.h"

namespace axon {

// ---------------------------------------------------------------------------
// Global autograd control (for inference without building a graph)
// ---------------------------------------------------------------------------
namespace {
bool g_grad_enabled = true;
}  // namespace

bool is_grad_enabled() {
    return g_grad_enabled;
}

void set_grad_enabled(bool enabled) {
    g_grad_enabled = enabled;
}

NoGradGuard::NoGradGuard() : prev_(g_grad_enabled) {
    g_grad_enabled = false;
}

NoGradGuard::~NoGradGuard() {
    g_grad_enabled = prev_;
}

// ---------------------------------------------------------------------------
// Tensor autograd methods (implemented here, where AutogradMeta is complete)
// ---------------------------------------------------------------------------
AutogradMeta& Tensor::ensure_meta() {
    if (!autograd_) {
        autograd_ = std::make_shared<AutogradMeta>();
    }
    return *autograd_;
}

AutogradMeta* Tensor::autograd_meta() const noexcept {
    return autograd_.get();
}

Tensor& Tensor::requires_grad_(bool req) {
    ensure_meta().requires_grad = req;
    return *this;
}

bool Tensor::requires_grad() const noexcept {
    return autograd_ && (autograd_->requires_grad || autograd_->grad_fn != nullptr);
}

bool Tensor::has_grad() const noexcept {
    return autograd_ && autograd_->has_grad;
}

const Tensor& Tensor::grad() const {
    if (!autograd_ || !autograd_->has_grad) {
        throw std::runtime_error("grad(): tensor has no accumulated gradient");
    }
    return autograd_->grad;
}

void Tensor::zero_grad() {
    if (autograd_) {
        autograd_->has_grad = false;
        autograd_->grad = Tensor{};
    }
}

void Tensor::set_grad_fn(const std::shared_ptr<Node>& fn) {
    ensure_meta().grad_fn = fn;
}

std::shared_ptr<Node> Tensor::grad_fn() const {
    return autograd_ ? autograd_->grad_fn : nullptr;
}

void Tensor::accumulate_grad(const Tensor& g) {
    AutogradMeta& m = ensure_meta();
    if (!m.has_grad) {
        m.grad = g.clone();
        m.has_grad = true;
    } else {
        m.grad = ops::add(m.grad, g);
    }
}

// ---------------------------------------------------------------------------
// Reverse-mode differentiation engine
// ---------------------------------------------------------------------------
void backward(Tensor& root) {
    // Seed the root gradient with 1 (dL/dL = 1).
    root.accumulate_grad(ops::ones_like(root));

    // Topological order (post-order) of the graph, starting at the root.
    std::vector<Tensor> topo;
    std::unordered_set<const AutogradMeta*> visited;

    std::function<void(const Tensor&)> dfs = [&](const Tensor& t) {
        const AutogradMeta* m = t.autograd_meta();
        if (m == nullptr || visited.count(m)) {
            return;
        }
        visited.insert(m);
        if (m->grad_fn) {
            for (const Tensor& inp : m->grad_fn->inputs) {
                dfs(inp);
            }
        }
        topo.push_back(t);
    };
    dfs(root);

    // Walk from the end (outputs) to the start (inputs), spreading gradients.
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        Tensor node_out = *it;
        AutogradMeta* m = node_out.autograd_meta();
        if (m == nullptr || !m->grad_fn || !m->has_grad) {
            continue;
        }
        std::vector<Tensor> in_grads = m->grad_fn->backward(m->grad);
        std::vector<Tensor>& ins = m->grad_fn->inputs;
        if (in_grads.size() != ins.size()) {
            throw std::runtime_error("backward: number of gradients != number of inputs");
        }
        for (std::size_t i = 0; i < ins.size(); ++i) {
            // Only propagate to inputs that participate in autograd.
            if (ins[i].requires_grad()) {
                ins[i].accumulate_grad(in_grads[i]);
            }
        }
    }
}

}  // namespace axon
