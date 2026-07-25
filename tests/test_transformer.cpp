#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <memory>

#include "axon/autograd.h"
#include "axon/functional.h"
#include "axon/nn.h"
#include "axon/ops.h"
#include "axon/optim.h"

using namespace axon;
namespace fn = axon::fn;
namespace ops = axon::ops;

namespace {
void check_grad(const std::function<Tensor(Tensor&)>& loss_fn, Tensor& x, float tol = 2e-2f) {
    Tensor loss = loss_fn(x);
    axon::backward(loss);
    Tensor analytic = x.grad().clone();
    const float h = 1e-3f;
    for (std::int64_t i = 0; i < x.numel(); ++i) {
        const float orig = x.data_ptr()[i];
        x.data_ptr()[i] = orig + h;
        const float lp = loss_fn(x).item();
        x.data_ptr()[i] = orig - h;
        const float lm = loss_fn(x).item();
        x.data_ptr()[i] = orig;
        EXPECT_NEAR(analytic.data_ptr()[i], (lp - lm) / (2.0f * h), tol) << "index " << i;
    }
}
}  // namespace

TEST(Transformer, LayerNormNormalizes) {
    // With gamma=1, beta=0, each row should have mean ~0 and variance ~1.
    Tensor x = Tensor::from_data({2, 4}, {1, 2, 3, 4, 10, 20, 30, 40});
    Tensor gamma = Tensor::ones({4});
    Tensor beta = Tensor::zeros({4});
    ops::LayerNormCache cache;
    Tensor y = ops::layernorm(x, gamma, beta, 1e-5f, &cache);
    for (std::int64_t i = 0; i < 2; ++i) {
        float mean = 0.0f;
        for (std::int64_t j = 0; j < 4; ++j) mean += y.at({i, j});
        mean /= 4.0f;
        EXPECT_NEAR(mean, 0.0f, 1e-4f);
    }
}

TEST(Transformer, GradCheckLayerNormX) {
    Tensor x = Tensor::from_data({2, 3}, {0.5f, -1.0f, 2.0f, 1.0f, 0.0f, -0.5f});
    x.requires_grad_(true);
    Tensor gamma = Tensor::from_data({3}, {1.2f, 0.8f, 1.0f});
    Tensor beta = Tensor::from_data({3}, {0.1f, -0.1f, 0.0f});
    Tensor w = Tensor::from_data({2, 3}, {1, 2, 3, 4, 5, 6});
    check_grad([&](Tensor& p) { return fn::sum(fn::mul(fn::layernorm(p, gamma, beta), w)); }, x);
}

TEST(Transformer, GradCheckLayerNormGamma) {
    Tensor x = Tensor::from_data({2, 3}, {0.5f, -1.0f, 2.0f, 1.0f, 0.0f, -0.5f});
    Tensor gamma = Tensor::from_data({3}, {1.2f, 0.8f, 1.0f});
    gamma.requires_grad_(true);
    Tensor beta = Tensor::zeros({3});
    Tensor w = Tensor::from_data({2, 3}, {1, 2, 3, 4, 5, 6});
    check_grad([&](Tensor& p) { return fn::sum(fn::mul(fn::layernorm(x, p, beta), w)); }, gamma);
}

TEST(Transformer, EmbeddingSelectsRows) {
    Tensor w = Tensor::from_data({3, 2}, {10, 11, 20, 21, 30, 31});
    Tensor idx = Tensor::from_data({2}, {2, 0});
    Tensor out = ops::embedding(w, idx);
    EXPECT_FLOAT_EQ(out.at({0, 0}), 30.0f);
    EXPECT_FLOAT_EQ(out.at({0, 1}), 31.0f);
    EXPECT_FLOAT_EQ(out.at({1, 0}), 10.0f);
}

TEST(Transformer, EmbeddingGradAccumulatesRepeatedIndex) {
    // Index 1 used twice -> its gradient must accumulate.
    Tensor w = Tensor::zeros({3, 2});
    w.requires_grad_(true);
    Tensor idx = Tensor::from_data({3}, {1, 1, 0});
    Tensor loss = fn::sum(fn::embedding(w, idx));
    axon::backward(loss);
    EXPECT_FLOAT_EQ(w.grad().at({1, 0}), 2.0f);  // used 2x
    EXPECT_FLOAT_EQ(w.grad().at({0, 0}), 1.0f);  // used 1x
    EXPECT_FLOAT_EQ(w.grad().at({2, 0}), 0.0f);  // not used
}

TEST(Transformer, AttentionShapeAndCausalMask) {
    const std::int64_t L = 3, d = 4;
    Tensor q = Tensor::randn({L, d}, 1);
    Tensor k = Tensor::randn({L, d}, 2);
    Tensor v = Tensor::randn({L, d}, 3);
    Tensor out = fn::scaled_dot_product_attention(q, k, v, /*causal=*/true);
    EXPECT_EQ(out.shape()[0], L);
    EXPECT_EQ(out.shape()[1], d);
}

TEST(Transformer, GradCheckAttention) {
    Tensor q = Tensor::randn({2, 3}, 5);
    q.requires_grad_(true);
    Tensor k = Tensor::randn({2, 3}, 6);
    Tensor v = Tensor::randn({2, 3}, 7);
    check_grad([&](Tensor& p) { return fn::sum(fn::scaled_dot_product_attention(p, k, v)); }, q);
}

TEST(Transformer, MultiHeadShapeAndParameters) {
    nn::MultiHeadAttention mha(8, /*num_heads=*/2, /*causal=*/true, 1);
    Tensor x = Tensor::randn({3, 8}, 5);
    Tensor y = mha.forward(x);
    EXPECT_EQ(y.shape()[0], 3);
    EXPECT_EQ(y.shape()[1], 8);
    EXPECT_EQ(mha.parameters().size(), 8u);  // 4 Linear x (W,b)
}

TEST(Transformer, AxonLMLogitsShape) {
    // vocab=10, dim=16, 2 heads, 2 layers.
    nn::AxonLM model(10, 16, 2, 2, 1);
    Tensor idx = Tensor::from_data({4}, {1, 2, 3, 4});  // sequence of 4 tokens
    Tensor logits = model.forward(idx);
    EXPECT_EQ(logits.shape()[0], 4);   // 1 distribution per position
    EXPECT_EQ(logits.shape()[1], 10);  // over the vocabulary
}

TEST(Transformer, AxonLMMemorizesSequence) {
    // The model should be able to "memorize" (overfit) a single short sequence:
    // predict the next token in [0,1,2,3] -> target [1,2,3,0].
    nn::AxonLM model(5, 32, 2, 2, 7);
    Tensor idx = Tensor::from_data({4}, {0, 1, 2, 3});
    Tensor targets = Tensor::from_data({4}, {1, 2, 3, 0});
    optim::Adam opt(model.parameters(), 0.01f);

    float first = 0.0f, last = 0.0f;
    for (int e = 0; e < 300; ++e) {
        Tensor logits = model.forward(idx);
        Tensor loss = fn::cross_entropy(logits, targets);
        if (e == 0) first = loss.item();
        last = loss.item();
        opt.zero_grad();
        backward(loss);
        opt.step();
    }
    EXPECT_LT(last, first * 0.2f);  // memorized (loss dropped a lot)
    EXPECT_LT(last, 0.1f);
}

TEST(Transformer, ForwardBatchEqualsIsolatedSequences) {
    // The forward_batch output for each sequence must be EQUAL to processing
    // that sequence alone with forward() (attention does not cross boundaries).
    nn::AxonLM model(6, 16, 2, 2, 3);
    const std::int64_t L = 4;
    Tensor s0 = Tensor::from_data({L}, {1, 2, 3, 4});
    Tensor s1 = Tensor::from_data({L}, {5, 0, 2, 1});

    Tensor out0 = model.forward(s0);  // (L, 6)
    Tensor out1 = model.forward(s1);

    Tensor idx = Tensor::from_data({2 * L}, {1, 2, 3, 4, 5, 0, 2, 1});
    Tensor batched = model.forward_batch(idx, /*batch=*/2, /*seq_len=*/L);  // (2L, 6)
    ASSERT_EQ(batched.shape()[0], 2 * L);
    ASSERT_EQ(batched.shape()[1], 6);

    for (std::int64_t i = 0; i < L; ++i) {
        for (std::int64_t j = 0; j < 6; ++j) {
            EXPECT_NEAR(batched.at({i, j}), out0.at({i, j}), 1e-4f);
            EXPECT_NEAR(batched.at({L + i, j}), out1.at({i, j}), 1e-4f);
        }
    }
}

TEST(Transformer, ForwardBatchTrains) {
    // Batch training: a single forward_batch call per step.
    nn::AxonLM model(5, 32, 2, 2, 7);
    const std::int64_t L = 4, B = 2;
    Tensor idx = Tensor::from_data({B * L}, {0, 1, 2, 3, 1, 2, 3, 0});
    Tensor tgt = Tensor::from_data({B * L}, {1, 2, 3, 0, 2, 3, 0, 1});
    optim::Adam opt(model.parameters(), 0.01f);

    float first = 0.0f, last = 0.0f;
    for (int e = 0; e < 300; ++e) {
        Tensor logits = model.forward_batch(idx, B, L);
        Tensor loss = fn::cross_entropy(logits, tgt);
        if (e == 0) first = loss.item();
        last = loss.item();
        opt.zero_grad();
        backward(loss);
        opt.step();
    }
    EXPECT_LT(last, first * 0.3f);
}

TEST(Transformer, KVCacheMatchesFullForward) {
    // Generating token by token with KV-cache must produce the SAME logits (at the
    // last position) as running the full forward over the entire sequence.
    nn::AxonLM model(7, 16, 2, 2, 5);
    std::vector<float> seq = {1, 3, 5, 2, 4};
    const std::int64_t L = static_cast<std::int64_t>(seq.size());

    // Full forward -> logits of the last position.
    Tensor full = model.forward(Tensor::from_data({L}, seq));  // (L, 7)
    const std::int64_t V = full.shape()[1];

    // KV-cache: processes one token at a time; keeps the logits of the last step.
    NoGradGuard ng;
    model.reset_cache();
    Tensor step_logits;
    for (std::int64_t t = 0; t < L; ++t) {
        step_logits = model.forward_step(static_cast<std::int64_t>(seq[t]), t);  // (1, V)
    }
    for (std::int64_t j = 0; j < V; ++j) {
        EXPECT_NEAR(step_logits.at({0, j}), full.at({L - 1, j}), 1e-3f) << "class " << j;
    }
}

TEST(Transformer, SelfAttentionTrains) {
    // Self-attention should be able to reduce a simple reconstruction loss.
    Tensor x = Tensor::randn({4, 8}, 1);
    Tensor target = Tensor::randn({4, 8}, 99);
    auto attn = std::make_shared<nn::SelfAttention>(8, /*causal=*/false, 3);
    optim::Adam opt(attn->parameters(), 0.01f);

    float first = 0.0f, last = 0.0f;
    for (int e = 0; e < 200; ++e) {
        Tensor loss = fn::mse_loss(attn->forward(x), target);
        if (e == 0) first = loss.item();
        last = loss.item();
        opt.zero_grad();
        backward(loss);
        opt.step();
    }
    EXPECT_LT(last, first);  // learned (loss dropped)
}
