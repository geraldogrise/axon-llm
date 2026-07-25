#pragma once

#include "axon/tensor.h"

// FUNCTIONAL layer: operations that compute the result AND build the autograd
// graph (they set grad_fn when some input requires a gradient). It is the analog of
// torch.nn.functional. Users normally use this (or the nn modules).
namespace axon::fn {

Tensor matmul(const Tensor& a, const Tensor& b);
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor sum(const Tensor& a);   // sums all elements -> scalar {1}
Tensor mean(const Tensor& a);  // mean of all elements -> scalar {1}

// Activations.
Tensor relu(const Tensor& a);
Tensor sigmoid(const Tensor& a);
Tensor tanh(const Tensor& a);
Tensor gelu(const Tensor& a);
Tensor softmax(const Tensor& a);      // 2D, over the last dimension
Tensor log_softmax(const Tensor& a);  // 2D, stable (log-sum-exp)

// Structural (Transformer).
Tensor transpose(const Tensor& a, std::int64_t dim0, std::int64_t dim1);
Tensor scale(const Tensor& a, float s);  // multiplies by a scalar (autograd)
Tensor layernorm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps = 1e-5f);
Tensor embedding(const Tensor& weight, const Tensor& idx);
// Attention: softmax(Q Kᵀ / sqrt(d) [+ causal mask]) V. Q,K,V are 2D (L, d).
Tensor scaled_dot_product_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                    bool causal = false);
// Column slicing and concatenation (for multi-head), with autograd.
Tensor narrow_cols(const Tensor& a, std::int64_t start, std::int64_t len);
Tensor cat_cols(const std::vector<Tensor>& parts);
// Row-wise analogs (for splitting/joining sequences of a batch), with autograd.
Tensor narrow_rows(const Tensor& a, std::int64_t start, std::int64_t len);
Tensor cat_rows(const std::vector<Tensor>& parts);

// Losses.
Tensor mse_loss(const Tensor& pred, const Tensor& target);
// Binary cross-entropy with logits (autograd). pred = logits, target in {0,1}.
Tensor bce_loss(const Tensor& pred, const Tensor& target);
// Huber loss (smooth L1) with autograd.
Tensor huber_loss(const Tensor& pred, const Tensor& target, float delta = 1.0f);
// 2D convolution with autograd. x:(N,Cin,H,W), weight:(Cout,Cin,K,K), bias:(Cout).
Tensor conv2d(const Tensor& x, const Tensor& weight, const Tensor& bias,
              int stride = 1, int pad = 0);
// Cross-entropy with logits; `targets` = class indices (shape {N}).
Tensor cross_entropy(const Tensor& logits, const Tensor& targets);

}  // namespace axon::fn

namespace axon {

// Syntactic sugar: operators that build the graph (call axon::fn).
Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);

}  // namespace axon
