#pragma once

#include <memory>
#include <vector>

#include "axon/tensor.h"

namespace axon {

// A node in the computation graph. Each differentiable operation creates a Node
// that stores its inputs and knows, given the output gradient, how to compute
// the gradient of each input (chain rule).
struct Node {
    std::vector<Tensor> inputs;  // inputs (share AutogradMeta with the originals)

    virtual ~Node() = default;

    // Receives the gradient of this node's output; returns one gradient per input.
    virtual std::vector<Tensor> backward(const Tensor& grad_output) = 0;

    [[nodiscard]] virtual const char* name() const { return "Node"; }
};

// Autograd metadata of a Tensor. Lives behind a shared_ptr so that all
// copies of the same Tensor see the SAME accumulated gradient.
struct AutogradMeta {
    bool requires_grad = false;        // is it a trainable leaf?
    bool has_grad = false;             // has it received a gradient yet?
    Tensor grad;                       // accumulated gradient (same shape as the data)
    std::shared_ptr<Node> grad_fn;     // node that produced the tensor (null if leaf)
};

// Runs reverse-mode differentiation starting from `root` (typically the loss,
// a scalar). Seeds grad(root) = 1 and propagates in reverse topological order,
// accumulating gradients in each leaf (parameter).
void backward(Tensor& root);

// ----- Autograd control (for inference) -----
// When disabled, operations do not build a graph (faster, less memory).
[[nodiscard]] bool is_grad_enabled();
void set_grad_enabled(bool enabled);

// RAII: disables autograd in the scope and restores it on exit (like torch.no_grad()).
struct NoGradGuard {
    NoGradGuard();
    ~NoGradGuard();
    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;

  private:
    bool prev_;
};

}  // namespace axon
