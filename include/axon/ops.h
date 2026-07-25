#pragma once

#include <vector>

#include "axon/tensor.h"

// Pure kernels over Tensors (computation only, WITHOUT an autograd graph).
// They serve as a reference/oracle and are the base upon which the functional
// layer (axon::fn, with autograd) is built.
namespace axon::ops {

// ----- Elementwise with broadcasting (NumPy-style rules) -----
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor neg(const Tensor& a);
Tensor mul_scalar(const Tensor& a, float s);

// ----- Linear algebra -----
// 2D matrix multiplication: (m,k) x (k,n) -> (m,n).
// Dispatches to the best available implementation (AVX2/OpenMP if present).
Tensor matmul(const Tensor& a, const Tensor& b);

// matmul variants, exposed for benchmarking and validation. All produce the
// same result; they differ only in performance.
namespace detail {
Tensor matmul_naive(const Tensor& a, const Tensor& b);    // ijk (slow reference)
Tensor matmul_ikj(const Tensor& a, const Tensor& b);      // reordered (cache-friendly)
Tensor matmul_blocked(const Tensor& a, const Tensor& b);  // tiling (cache blocking)
Tensor matmul_omp(const Tensor& a, const Tensor& b);      // ikj + OpenMP (multithread)
Tensor matmul_avx2(const Tensor& a, const Tensor& b);     // SIMD AVX2 + OpenMP
Tensor matmul_avx512(const Tensor& a, const Tensor& b);   // SIMD AVX-512 + OpenMP (if available)
}  // namespace detail

// ----- Activations (forward + gradient kernels) -----
Tensor relu(const Tensor& a);
Tensor relu_mask(const Tensor& a);  // 1.0 where a > 0, else 0.0 (ReLU derivative)

Tensor sigmoid(const Tensor& a);
Tensor sigmoid_grad(const Tensor& y, const Tensor& g);  // g * y * (1 - y), y = sigmoid(x)

Tensor tanh(const Tensor& a);
Tensor tanh_grad(const Tensor& y, const Tensor& g);  // g * (1 - y^2), y = tanh(x)

Tensor gelu(const Tensor& a);
Tensor gelu_grad(const Tensor& x, const Tensor& g);  // g * d/dx GELU(x)

// FUSED kernels (bias + activation in a single pass, no intermediate tensor).
// x: (N, D); bias: (D,) added by broadcasting before the activation.
Tensor bias_relu(const Tensor& x, const Tensor& bias);
Tensor bias_gelu(const Tensor& x, const Tensor& bias);

// Softmax over the last dimension of a 2D tensor (batch, features), stable.
Tensor softmax(const Tensor& a);
Tensor softmax_grad(const Tensor& y, const Tensor& g);  // y = softmax(x)

// Stable log-softmax (log-sum-exp) over the last dimension of a 2D tensor.
Tensor log_softmax(const Tensor& a);
Tensor log_softmax_grad(const Tensor& y, const Tensor& g);  // y = log_softmax(x)

// Cross-entropy with logits. `targets` are class indices (shape {N}, floats).
Tensor cross_entropy(const Tensor& logits, const Tensor& targets);       // -> scalar {1}
Tensor cross_entropy_grad(const Tensor& logits, const Tensor& targets);  // -> dlogits (N,C)

// Binary cross-entropy WITH logits (numerically stable). Same shape pred/target.
Tensor bce_with_logits(const Tensor& pred, const Tensor& target);        // -> scalar {1}
Tensor bce_with_logits_grad(const Tensor& pred, const Tensor& target);   // = (sigmoid(x)-y)/N

// Huber loss (smooth L1). Quadratic within delta, linear outside.
Tensor huber(const Tensor& pred, const Tensor& target, float delta);     // -> scalar {1}
Tensor huber_grad(const Tensor& pred, const Tensor& target, float delta);

// 2D convolution. x:(N,Cin,H,W), weight:(Cout,Cin,K,K), bias:(Cout) -> (N,Cout,Hout,Wout).
Tensor conv2d(const Tensor& x, const Tensor& weight, const Tensor& bias, int stride, int pad);
// Backward: returns {dx, dweight, dbias} given the output gradient g.
std::vector<Tensor> conv2d_backward(const Tensor& x, const Tensor& weight, const Tensor& g,
                                    int stride, int pad);

// ----- LayerNorm (normalizes over the last dimension of a 2D tensor) -----
// Stores what the backward needs: xhat (normalized) and rstd (1/sqrt(var+eps)).
struct LayerNormCache {
    Tensor xhat;
    Tensor rstd;
};
Tensor layernorm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps,
                 LayerNormCache* cache);
void layernorm_backward(const Tensor& dy, const Tensor& xhat, const Tensor& rstd,
                        const Tensor& gamma, Tensor& dx, Tensor& dgamma, Tensor& dbeta);

// ----- Embedding (lookup table: indices -> vectors) -----
// weight: (num_embeddings, dim); idx: (N,) float indices -> output (N, dim).
Tensor embedding(const Tensor& weight, const Tensor& idx);
Tensor embedding_backward(const Tensor& idx, const Tensor& dy, std::int64_t num_embeddings);

// Causal mask (L,L): 0 on the diagonal and below, -1e9 above (prevents seeing the future).
Tensor causal_mask(std::int64_t L);

// ----- Shape operations (for multi-head attention) -----
// Slices columns [start, start+len) from a 2D tensor (rows, C) -> (rows, len).
Tensor slice_cols(const Tensor& a, std::int64_t start, std::int64_t len);
// Backward of the slice: scatters g back into a zeroed (rows, full_cols) tensor.
Tensor slice_cols_backward(const Tensor& g, std::int64_t start, std::int64_t full_cols);
// Concatenates 2D tensors along the columns: [(rows,c1),(rows,c2),...] -> (rows, sum ci).
Tensor concat_cols(const std::vector<Tensor>& parts);

// ROW-wise analogs (for splitting/joining sequences of a batch).
Tensor slice_rows(const Tensor& a, std::int64_t start, std::int64_t len);
Tensor slice_rows_backward(const Tensor& g, std::int64_t start, std::int64_t full_rows);
Tensor concat_rows(const std::vector<Tensor>& parts);

// Sinusoidal positional encoding (constant): (L, D).
Tensor positional_encoding(std::int64_t L, std::int64_t D);

// ----- int8 quantization (symmetric per tensor) -----
// Quantizes to the int8 grid [-127,127] (values stored as float); also returns
// the scale (scale = max|t| / 127). dequantize_int8 undoes it (q * scale).
Tensor quantize_int8(const Tensor& t, float& scale_out);
Tensor dequantize_int8(const Tensor& q, float scale);

// ----- Reductions -----
Tensor sum_all(const Tensor& a);   // sum of all elements -> shape {1}
Tensor mean_all(const Tensor& a);  // mean of all elements -> shape {1}

// Reductions along an axis (keepdim optional).
Tensor sum_axis(const Tensor& a, std::int64_t axis, bool keepdim = false);
Tensor mean_axis(const Tensor& a, std::int64_t axis, bool keepdim = false);
Tensor max_axis(const Tensor& a, std::int64_t axis, bool keepdim = false);
Tensor min_axis(const Tensor& a, std::int64_t axis, bool keepdim = false);

// ----- Gradient utilities -----
// Reduces `grad` (by summing) until it matches `target_shape`, undoing a
// broadcasting done in the forward pass. E.g.: (m,n) -> (n,) by summing axis 0.
Tensor sum_to(const Tensor& grad, const Shape& target_shape);
// Tensor of the same shape as `t`, filled with 1.
Tensor ones_like(const Tensor& t);

}  // namespace axon::ops
