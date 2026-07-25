#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "axon/tensor.h"

namespace axon::nn {

// Base building block of a network. Every module transforms a Tensor and exposes
// its trainable parameters (for the optimizer).
class Module {
  public:
    virtual ~Module() = default;
    virtual Tensor forward(const Tensor& x) = 0;
    virtual std::vector<Tensor> parameters() = 0;

    Tensor operator()(const Tensor& x) { return forward(x); }
};

// Weight initialization scheme.
enum class Init { He, Xavier, Uniform };

// Fully connected layer: y = x @ W + b.
// W has shape (in_features, out_features); b has shape (out_features).
class Linear : public Module {
  public:
    Linear(std::int64_t in_features, std::int64_t out_features, std::uint64_t seed = 42,
           Init init = Init::He);
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor> parameters() override { return {weight_, bias_}; }

    [[nodiscard]] Tensor& weight() { return weight_; }
    [[nodiscard]] Tensor& bias() { return bias_; }

  private:
    Tensor weight_;
    Tensor bias_;
};

// Activations as modules (no parameters).
class ReLU : public Module {
  public:
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor> parameters() override { return {}; }
};

class Sigmoid : public Module {
  public:
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor> parameters() override { return {}; }
};

class Tanh : public Module {
  public:
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor> parameters() override { return {}; }
};

class GELU : public Module {
  public:
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor> parameters() override { return {}; }
};

// Layer normalization (over the last dimension), with trainable gamma/beta.
class LayerNorm : public Module {
  public:
    explicit LayerNorm(std::int64_t dim, float eps = 1e-5f);
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor> parameters() override { return {gamma_, beta_}; }

  private:
    Tensor gamma_;
    Tensor beta_;
    float eps_;
};

// Embedding table: indices (N,) -> vectors (N, dim).
class Embedding : public Module {
  public:
    Embedding(std::int64_t num_embeddings, std::int64_t dim, std::uint64_t seed = 42);
    Tensor forward(const Tensor& idx) override;
    std::vector<Tensor> parameters() override { return {weight_}; }

  private:
    Tensor weight_;
};

// Single-head self-attention: project Q,K,V, apply attention and project the output.
class SelfAttention : public Module {
  public:
    SelfAttention(std::int64_t dim, bool causal = false, std::uint64_t seed = 42);
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor> parameters() override;

  private:
    Linear wq_, wk_, wv_, wo_;
    bool causal_;
};

// Multi-head attention: split the dimension into `num_heads` heads, apply attention
// in each one and concatenate the results.
class MultiHeadAttention : public Module {
  public:
    MultiHeadAttention(std::int64_t dim, std::int64_t num_heads, bool causal = false,
                       std::uint64_t seed = 42);
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor> parameters() override;
    // Batch mode: the input (B*L, D) is treated as B sequences of length L.
    // 0 = a single sequence (all rows together).
    void set_seq_len(std::int64_t seq_len) { seq_len_ = seq_len; }
    // Incremental step (KV-cache): x_new (1,D); appends K,V to the cache and attends
    // to the whole past. Returns (1,D).
    Tensor forward_step(const Tensor& x_new, Tensor& k_cache, Tensor& v_cache);

  private:
    Linear wq_, wk_, wv_, wo_;
    std::int64_t num_heads_;
    std::int64_t head_dim_;
    bool causal_;
    std::int64_t seq_len_ = 0;
};

// Decoder-only Transformer block (pre-norm, GPT-2 style):
//   x = x + attn(LN(x));  x = x + FFN(LN(x))
class TransformerBlock : public Module {
  public:
    TransformerBlock(std::int64_t dim, std::int64_t num_heads, bool causal = true,
                     std::uint64_t seed = 42, std::int64_t ff_mult = 4);
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor> parameters() override;
    void set_seq_len(std::int64_t seq_len) { attn_.set_seq_len(seq_len); }
    // Incremental step (KV-cache) for a single new token (1,D).
    Tensor forward_step(const Tensor& x_new, Tensor& k_cache, Tensor& v_cache);

  private:
    LayerNorm ln1_, ln2_;
    MultiHeadAttention attn_;
    Linear ff1_, ff2_;
};

// AxonLM: decoder-only language model (mini generative Transformer).
// forward(idx): token indices (L,) -> logits (L, vocab_size).
class AxonLM : public Module {
  public:
    AxonLM(std::int64_t vocab_size, std::int64_t dim, std::int64_t num_heads,
           std::int64_t num_layers, std::uint64_t seed = 42);
    Tensor forward(const Tensor& idx) override;  // 1 sequence: idx (L,) -> (L, vocab)
    // Batch: idx (batch*seq_len,) flattened -> logits (batch*seq_len, vocab).
    Tensor forward_batch(const Tensor& idx, std::int64_t batch, std::int64_t seq_len);
    std::vector<Tensor> parameters() override;

    // ----- Incremental generation with KV-cache -----
    void reset_cache();  // clear the cache (new sequence)
    // Process a single token at position `position`; uses/updates the cache.
    // Returns the logits (1, vocab). Run inside no_grad.
    Tensor forward_step(std::int64_t token_id, std::int64_t position);

    [[nodiscard]] std::int64_t vocab_size() const { return vocab_size_; }

  private:
    Embedding tok_emb_;
    std::vector<std::shared_ptr<TransformerBlock>> blocks_;
    LayerNorm lnf_;
    Linear head_;
    std::int64_t dim_;
    std::int64_t vocab_size_;
    std::vector<Tensor> k_cache_, v_cache_;  // one per block
};

// Dropout: during training, zeroes each element with probability p and scales the
// rest by 1/(1-p). At inference (no_grad) it becomes the identity.
class Dropout : public Module {
  public:
    explicit Dropout(float p = 0.5f, std::uint64_t seed = 1234);
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor> parameters() override { return {}; }

  private:
    float p_;
    std::uint64_t state_;  // RNG state (advances on each forward)
};

// Simple RNN (Elman): h_t = tanh(W_ih x_t + W_hh h_{t-1}).
// Takes (L, in) -> returns (L, hidden).
class RNN : public Module {
  public:
    RNN(std::int64_t in_features, std::int64_t hidden, std::uint64_t seed = 42);
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor> parameters() override;

  private:
    Linear ih_, hh_;
    std::int64_t hidden_;
};

// LSTM: input/forget/candidate/output gates with a memory cell.
// Takes (L, in) -> returns (L, hidden).
class LSTM : public Module {
  public:
    LSTM(std::int64_t in_features, std::int64_t hidden, std::uint64_t seed = 42);
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor> parameters() override;

  private:
    Linear ih_, hh_;  // project to 4*hidden (i, f, g, o)
    std::int64_t hidden_;
};

// Chains modules: the output of one is the input of the next.
class Sequential : public Module {
  public:
    Sequential() = default;
    explicit Sequential(std::vector<std::shared_ptr<Module>> layers)
        : layers_(std::move(layers)) {}

    void add(std::shared_ptr<Module> layer) { layers_.push_back(std::move(layer)); }
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor> parameters() override;

  private:
    std::vector<std::shared_ptr<Module>> layers_;
};

}  // namespace axon::nn
