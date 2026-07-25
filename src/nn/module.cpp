#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>

#include "axon/autograd.h"
#include "axon/functional.h"
#include "axon/nn.h"
#include "axon/ops.h"

namespace axon::nn {

Linear::Linear(std::int64_t in_features, std::int64_t out_features, std::uint64_t seed,
               Init init) {
    const float fan_in = static_cast<float>(in_features);
    const float fan_out = static_cast<float>(out_features);
    switch (init) {
        case Init::He:  // good for ReLU: std = sqrt(2 / fan_in)
            weight_ = Tensor::randn({in_features, out_features}, seed, 0.0f,
                                    std::sqrt(2.0f / fan_in));
            break;
        case Init::Xavier: {  // Glorot: std = sqrt(2 / (fan_in + fan_out))
            const float s = std::sqrt(2.0f / (fan_in + fan_out));
            weight_ = Tensor::randn({in_features, out_features}, seed, 0.0f, s);
            break;
        }
        case Init::Uniform: {  // uniform [-limit, limit], limit = sqrt(1/fan_in)
            const float limit = std::sqrt(1.0f / fan_in);
            weight_ = Tensor::rand({in_features, out_features}, seed, -limit, limit);
            break;
        }
    }
    weight_.requires_grad_(true);
    bias_ = Tensor::zeros({out_features});
    bias_.requires_grad_(true);
}

Tensor Linear::forward(const Tensor& x) {
    // x: (batch, in) @ W: (in, out) -> (batch, out); add the bias (broadcast).
    return fn::add(fn::matmul(x, weight_), bias_);
}

Tensor ReLU::forward(const Tensor& x) {
    return fn::relu(x);
}

Tensor Sigmoid::forward(const Tensor& x) {
    return fn::sigmoid(x);
}

Tensor Tanh::forward(const Tensor& x) {
    return fn::tanh(x);
}

Tensor GELU::forward(const Tensor& x) {
    return fn::gelu(x);
}

LayerNorm::LayerNorm(std::int64_t dim, float eps) : eps_(eps) {
    gamma_ = Tensor::ones({dim});
    gamma_.requires_grad_(true);
    beta_ = Tensor::zeros({dim});
    beta_.requires_grad_(true);
}

Tensor LayerNorm::forward(const Tensor& x) {
    return fn::layernorm(x, gamma_, beta_, eps_);
}

Embedding::Embedding(std::int64_t num_embeddings, std::int64_t dim, std::uint64_t seed) {
    // Small init (std=0.02), as in GPTs.
    weight_ = Tensor::randn({num_embeddings, dim}, seed, 0.0f, 0.02f);
    weight_.requires_grad_(true);
}

Tensor Embedding::forward(const Tensor& idx) {
    return fn::embedding(weight_, idx);
}

SelfAttention::SelfAttention(std::int64_t dim, bool causal, std::uint64_t seed)
    : wq_(dim, dim, seed), wk_(dim, dim, seed + 1), wv_(dim, dim, seed + 2),
      wo_(dim, dim, seed + 3), causal_(causal) {}

Tensor SelfAttention::forward(const Tensor& x) {
    Tensor q = wq_.forward(x);
    Tensor k = wk_.forward(x);
    Tensor v = wv_.forward(x);
    Tensor a = fn::scaled_dot_product_attention(q, k, v, causal_);
    return wo_.forward(a);
}

std::vector<Tensor> SelfAttention::parameters() {
    std::vector<Tensor> params;
    Linear* layers[] = {&wq_, &wk_, &wv_, &wo_};
    for (Linear* m : layers) {
        for (Tensor& p : m->parameters()) {
            params.push_back(p);
        }
    }
    return params;
}

namespace {
// Appends a module's parameters to an accumulator vector.
void append_params(std::vector<Tensor>& dst, Module& m) {
    for (Tensor& p : m.parameters()) {
        dst.push_back(p);
    }
}
}  // namespace

MultiHeadAttention::MultiHeadAttention(std::int64_t dim, std::int64_t num_heads, bool causal,
                                       std::uint64_t seed)
    : wq_(dim, dim, seed), wk_(dim, dim, seed + 1), wv_(dim, dim, seed + 2),
      wo_(dim, dim, seed + 3), num_heads_(num_heads), head_dim_(dim / num_heads),
      causal_(causal) {
    if (dim % num_heads != 0) {
        throw std::invalid_argument("MultiHeadAttention: dim must be divisible by num_heads");
    }
}

namespace {
// Multi-head attention over ONE sequence (L, D): split into heads and concatenate.
Tensor mha_one_sequence(const Tensor& q, const Tensor& k, const Tensor& v,
                        std::int64_t num_heads, std::int64_t head_dim, bool causal) {
    std::vector<Tensor> heads;
    heads.reserve(static_cast<std::size_t>(num_heads));
    for (std::int64_t h = 0; h < num_heads; ++h) {
        const std::int64_t s = h * head_dim;
        Tensor qh = fn::narrow_cols(q, s, head_dim);
        Tensor kh = fn::narrow_cols(k, s, head_dim);
        Tensor vh = fn::narrow_cols(v, s, head_dim);
        heads.push_back(fn::scaled_dot_product_attention(qh, kh, vh, causal));
    }
    return fn::cat_cols(heads);
}
}  // namespace

Tensor MultiHeadAttention::forward(const Tensor& x) {
    // Projections done once over all rows (one large, efficient matmul).
    Tensor q = wq_.forward(x);
    Tensor k = wk_.forward(x);
    Tensor v = wv_.forward(x);

    const std::int64_t rows = x.shape()[0];
    const std::int64_t L = (seq_len_ > 0) ? seq_len_ : rows;
    const std::int64_t B = rows / L;

    if (B <= 1) {
        return wo_.forward(mha_one_sequence(q, k, v, num_heads_, head_dim_, causal_));
    }
    // Batch: split per sequence (attention doesn't cross boundaries), then join.
    std::vector<Tensor> seqs;
    seqs.reserve(static_cast<std::size_t>(B));
    for (std::int64_t b = 0; b < B; ++b) {
        const std::int64_t r = b * L;
        Tensor qb = fn::narrow_rows(q, r, L);
        Tensor kb = fn::narrow_rows(k, r, L);
        Tensor vb = fn::narrow_rows(v, r, L);
        seqs.push_back(mha_one_sequence(qb, kb, vb, num_heads_, head_dim_, causal_));
    }
    return wo_.forward(fn::cat_rows(seqs));
}

Tensor MultiHeadAttention::forward_step(const Tensor& x_new, Tensor& k_cache, Tensor& v_cache) {
    // x_new: (1, D). Project Q,K,V of the new token and append K,V to the cache.
    Tensor q = wq_.forward(x_new);
    Tensor k = wk_.forward(x_new);
    Tensor v = wv_.forward(x_new);
    if (k_cache.numel() == 0) {
        k_cache = k.clone();
        v_cache = v.clone();
    } else {
        k_cache = ops::concat_rows({k_cache, k});  // (T, D)
        v_cache = ops::concat_rows({v_cache, v});
    }
    // Attend the new token (1, dh) to the whole past (T, dh), per head.
    std::vector<Tensor> heads;
    heads.reserve(static_cast<std::size_t>(num_heads_));
    for (std::int64_t h = 0; h < num_heads_; ++h) {
        const std::int64_t s = h * head_dim_;
        Tensor qh = fn::narrow_cols(q, s, head_dim_);
        Tensor kh = fn::narrow_cols(k_cache, s, head_dim_);
        Tensor vh = fn::narrow_cols(v_cache, s, head_dim_);
        heads.push_back(fn::scaled_dot_product_attention(qh, kh, vh, /*causal=*/false));
    }
    return wo_.forward(fn::cat_cols(heads));
}

std::vector<Tensor> MultiHeadAttention::parameters() {
    std::vector<Tensor> params;
    Linear* layers[] = {&wq_, &wk_, &wv_, &wo_};
    for (Linear* m : layers) {
        append_params(params, *m);
    }
    return params;
}

TransformerBlock::TransformerBlock(std::int64_t dim, std::int64_t num_heads, bool causal,
                                   std::uint64_t seed, std::int64_t ff_mult)
    : ln1_(dim), ln2_(dim), attn_(dim, num_heads, causal, seed),
      ff1_(dim, dim * ff_mult, seed + 10), ff2_(dim * ff_mult, dim, seed + 11) {}

Tensor TransformerBlock::forward(const Tensor& x) {
    // Residual connections with pre-normalization (GPT-2 style).
    Tensor a = attn_.forward(ln1_.forward(x));
    Tensor x1 = fn::add(x, a);
    Tensor f = ff2_.forward(fn::gelu(ff1_.forward(ln2_.forward(x1))));
    return fn::add(x1, f);
}

Tensor TransformerBlock::forward_step(const Tensor& x_new, Tensor& k_cache, Tensor& v_cache) {
    Tensor a = attn_.forward_step(ln1_.forward(x_new), k_cache, v_cache);
    Tensor x1 = fn::add(x_new, a);
    Tensor f = ff2_.forward(fn::gelu(ff1_.forward(ln2_.forward(x1))));
    return fn::add(x1, f);
}

std::vector<Tensor> TransformerBlock::parameters() {
    std::vector<Tensor> params;
    append_params(params, ln1_);
    append_params(params, attn_);
    append_params(params, ln2_);
    append_params(params, ff1_);
    append_params(params, ff2_);
    return params;
}

AxonLM::AxonLM(std::int64_t vocab_size, std::int64_t dim, std::int64_t num_heads,
               std::int64_t num_layers, std::uint64_t seed)
    : tok_emb_(vocab_size, dim, seed), lnf_(dim), head_(dim, vocab_size, seed + 1), dim_(dim),
      vocab_size_(vocab_size) {
    for (std::int64_t i = 0; i < num_layers; ++i) {
        blocks_.push_back(std::make_shared<TransformerBlock>(
            dim, num_heads, /*causal=*/true, seed + 100 + static_cast<std::uint64_t>(i) * 20));
    }
}

Tensor AxonLM::forward(const Tensor& idx) {
    Tensor x = tok_emb_.forward(idx);  // (L, dim)
    const std::int64_t L = x.shape()[0];
    x = fn::add(x, ops::positional_encoding(L, dim_));  // + position (constant)
    for (auto& block : blocks_) {
        x = block->forward(x);
    }
    x = lnf_.forward(x);
    return head_.forward(x);  // (L, vocab_size)
}

Tensor AxonLM::forward_batch(const Tensor& idx, std::int64_t batch, std::int64_t seq_len) {
    Tensor x = tok_emb_.forward(idx);  // (batch*seq_len, dim)
    // Positional encoding replicated per sequence (constant).
    Tensor pe1 = ops::positional_encoding(seq_len, dim_);
    std::vector<Tensor> tiles(static_cast<std::size_t>(batch), pe1);
    x = fn::add(x, ops::concat_rows(tiles));

    for (auto& block : blocks_) {
        block->set_seq_len(seq_len);
        x = block->forward(x);
    }
    x = lnf_.forward(x);
    Tensor out = head_.forward(x);  // (batch*seq_len, vocab)
    for (auto& block : blocks_) {
        block->set_seq_len(0);  // restore single-sequence mode
    }
    return out;
}

void AxonLM::reset_cache() {
    k_cache_.assign(blocks_.size(), Tensor{});
    v_cache_.assign(blocks_.size(), Tensor{});
}

Tensor AxonLM::forward_step(std::int64_t token_id, std::int64_t position) {
    if (k_cache_.size() != blocks_.size()) {
        reset_cache();
    }
    Tensor idx = Tensor::from_data({1}, {static_cast<float>(token_id)});
    Tensor x = tok_emb_.forward(idx);  // (1, dim)
    // Positional encoding of the absolute position (row `position`).
    Tensor pe = ops::positional_encoding(position + 1, dim_);
    x = fn::add(x, ops::slice_rows(pe, position, 1));
    for (std::size_t i = 0; i < blocks_.size(); ++i) {
        x = blocks_[i]->forward_step(x, k_cache_[i], v_cache_[i]);
    }
    x = lnf_.forward(x);
    return head_.forward(x);  // (1, vocab)
}

std::vector<Tensor> AxonLM::parameters() {
    std::vector<Tensor> params;
    append_params(params, tok_emb_);
    for (auto& block : blocks_) {
        append_params(params, *block);
    }
    append_params(params, lnf_);
    append_params(params, head_);
    return params;
}

Dropout::Dropout(float p, std::uint64_t seed) : p_(p), state_(seed) {}

Tensor Dropout::forward(const Tensor& x) {
    // At inference (autograd disabled) or p<=0: identity.
    if (!is_grad_enabled() || p_ <= 0.0f) {
        return x;
    }
    const Tensor cx = x.contiguous();
    Tensor mask = Tensor::zeros(cx.shape());
    const float scale = 1.0f / (1.0f - p_);
    std::mt19937_64 gen(state_);
    state_ = gen();  // advance the seed for the next forward
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    float* pm = mask.data_ptr();
    const std::int64_t nelem = mask.numel();
    for (std::int64_t i = 0; i < nelem; ++i) {
        pm[i] = (uni(gen) < p_) ? 0.0f : scale;
    }
    return fn::mul(x, mask);  // mask is constant -> grad passes through scaled by the mask
}

RNN::RNN(std::int64_t in_features, std::int64_t hidden, std::uint64_t seed)
    : ih_(in_features, hidden, seed), hh_(hidden, hidden, seed + 1), hidden_(hidden) {}

Tensor RNN::forward(const Tensor& x) {
    const std::int64_t L = x.shape()[0];
    Tensor h = Tensor::zeros({1, hidden_});
    std::vector<Tensor> outs;
    outs.reserve(static_cast<std::size_t>(L));
    for (std::int64_t t = 0; t < L; ++t) {
        Tensor x_t = fn::narrow_rows(x, t, 1);  // (1, in)
        h = fn::tanh(fn::add(ih_.forward(x_t), hh_.forward(h)));
        outs.push_back(h);
    }
    return fn::cat_rows(outs);  // (L, hidden)
}

std::vector<Tensor> RNN::parameters() {
    std::vector<Tensor> p;
    append_params(p, ih_);
    append_params(p, hh_);
    return p;
}

LSTM::LSTM(std::int64_t in_features, std::int64_t hidden, std::uint64_t seed)
    : ih_(in_features, hidden * 4, seed), hh_(hidden, hidden * 4, seed + 1), hidden_(hidden) {}

Tensor LSTM::forward(const Tensor& x) {
    const std::int64_t L = x.shape()[0];
    const std::int64_t H = hidden_;
    Tensor h = Tensor::zeros({1, H});
    Tensor c = Tensor::zeros({1, H});
    std::vector<Tensor> outs;
    outs.reserve(static_cast<std::size_t>(L));
    for (std::int64_t t = 0; t < L; ++t) {
        Tensor x_t = fn::narrow_rows(x, t, 1);
        Tensor z = fn::add(ih_.forward(x_t), hh_.forward(h));  // (1, 4H)
        Tensor i = fn::sigmoid(fn::narrow_cols(z, 0, H));
        Tensor f = fn::sigmoid(fn::narrow_cols(z, H, H));
        Tensor g = fn::tanh(fn::narrow_cols(z, 2 * H, H));
        Tensor o = fn::sigmoid(fn::narrow_cols(z, 3 * H, H));
        c = fn::add(fn::mul(f, c), fn::mul(i, g));
        h = fn::mul(o, fn::tanh(c));
        outs.push_back(h);
    }
    return fn::cat_rows(outs);  // (L, hidden)
}

std::vector<Tensor> LSTM::parameters() {
    std::vector<Tensor> p;
    append_params(p, ih_);
    append_params(p, hh_);
    return p;
}

Tensor Sequential::forward(const Tensor& x) {
    Tensor out = x;
    for (auto& layer : layers_) {
        out = layer->forward(out);
    }
    return out;
}

std::vector<Tensor> Sequential::parameters() {
    std::vector<Tensor> params;
    for (auto& layer : layers_) {
        for (Tensor& p : layer->parameters()) {
            params.push_back(p);
        }
    }
    return params;
}

}  // namespace axon::nn
