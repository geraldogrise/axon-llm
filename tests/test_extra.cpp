#include <gtest/gtest.h>

#include <cmath>
#include <functional>

#include "axon/autograd.h"
#include "axon/functional.h"
#include "axon/nn.h"
#include "axon/ops.h"
#include "axon/optim.h"

using namespace axon;
namespace fn = axon::fn;
namespace ops = axon::ops;

TEST(Extra, RandWithinInterval) {
    Tensor t = Tensor::rand({100}, 7, -2.0f, 3.0f);
    for (std::int64_t i = 0; i < t.numel(); ++i) {
        EXPECT_GE(t.data_ptr()[i], -2.0f);
        EXPECT_LT(t.data_ptr()[i], 3.0f);
    }
}

TEST(Extra, PermuteIsView) {
    Tensor t = Tensor::from_data({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor p = t.permute({1, 0});  // (3,2)
    EXPECT_EQ(p.shape()[0], 3);
    EXPECT_EQ(p.shape()[1], 2);
    EXPECT_FLOAT_EQ((p.at({0, 1})), (t.at({1, 0})));
    EXPECT_FLOAT_EQ((p.at({2, 1})), (t.at({1, 2})));
}

TEST(Extra, SliceCutsDimension) {
    Tensor t = Tensor::from_data({4, 2}, {0, 1, 2, 3, 4, 5, 6, 7});
    Tensor s = t.slice(0, 1, 2);  // rows 1..2 -> (2,2)
    EXPECT_EQ(s.shape()[0], 2);
    EXPECT_FLOAT_EQ((s.at({0, 0})), 2.0f);
    EXPECT_FLOAT_EQ((s.at({1, 1})), 5.0f);
}

TEST(Extra, ReductionsByAxis) {
    Tensor t = Tensor::from_data({2, 3}, {1, 2, 3, 4, 5, 6});
    // sum over axis 0 -> [5, 7, 9]
    Tensor s0 = ops::sum_axis(t, 0);
    EXPECT_FLOAT_EQ(s0.at({0}), 5.0f);
    EXPECT_FLOAT_EQ(s0.at({2}), 9.0f);
    // mean over axis 1 -> [2, 5]
    Tensor m1 = ops::mean_axis(t, 1);
    EXPECT_FLOAT_EQ(m1.at({0}), 2.0f);
    EXPECT_FLOAT_EQ(m1.at({1}), 5.0f);
    // max over axis 1 -> [3, 6]; min -> [1, 4]
    EXPECT_FLOAT_EQ(ops::max_axis(t, 1).at({1}), 6.0f);
    EXPECT_FLOAT_EQ(ops::min_axis(t, 1).at({0}), 1.0f);
}

TEST(Extra, LogSoftmaxValueAndGradCheck) {
    // log_softmax(x)_j = x_j - logsumexp(x). exp(log_softmax) sums to 1.
    Tensor x = Tensor::from_data({1, 3}, {1.0f, 2.0f, 3.0f});
    Tensor y = ops::log_softmax(x);
    float s = 0.0f;
    for (std::int64_t j = 0; j < 3; ++j) s += std::exp(y.at({0, j}));
    EXPECT_NEAR(s, 1.0f, 1e-5f);

    // Gradient check of sum(w * log_softmax(x)).
    Tensor xr = Tensor::from_data({2, 3}, {0.5f, -1.0f, 2.0f, 1.0f, 0.0f, -0.5f});
    xr.requires_grad_(true);
    Tensor w = Tensor::from_data({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor loss = fn::sum(fn::mul(fn::log_softmax(xr), w));
    axon::backward(loss);
    Tensor analytic = xr.grad().clone();
    const float h = 1e-3f;
    for (std::int64_t i = 0; i < xr.numel(); ++i) {
        const float o = xr.data_ptr()[i];
        xr.data_ptr()[i] = o + h;
        const float lp = fn::sum(fn::mul(fn::log_softmax(xr), w)).item();
        xr.data_ptr()[i] = o - h;
        const float lm = fn::sum(fn::mul(fn::log_softmax(xr), w)).item();
        xr.data_ptr()[i] = o;
        EXPECT_NEAR(analytic.data_ptr()[i], (lp - lm) / (2 * h), 2e-2f);
    }
}

TEST(Extra, XavierInitDiffersFromHe) {
    nn::Linear he(64, 32, 1, nn::Init::He);
    nn::Linear xa(64, 32, 1, nn::Init::Xavier);
    // Same seed, different scales -> different weights.
    EXPECT_NE(he.weight().at({0, 0}), xa.weight().at({0, 0}));
}

TEST(Extra, SetLrAffectsOptimizer) {
    Tensor w = Tensor::from_data({2}, {1, 2});
    optim::SGD opt({w}, 0.1f);
    EXPECT_FLOAT_EQ(opt.lr(), 0.1f);
    opt.set_lr(0.05f);
    EXPECT_FLOAT_EQ(opt.lr(), 0.05f);
}

TEST(Extra, AdamStateDictRoundTrip) {
    // Trains a few steps, captures the state, and verifies that resuming reproduces the step.
    auto make = [] {
        Tensor w = Tensor::from_data({3}, {1, 2, 3});
        w.requires_grad_(true);
        return w;
    };
    Tensor w1 = make();
    optim::Adam o1({w1}, 0.1f);
    for (int i = 0; i < 3; ++i) {
        Tensor loss = fn::sum(fn::mul(w1, w1));  // grad = 2w
        w1.zero_grad();
        axon::backward(loss);
        o1.step();
    }
    std::vector<Tensor> state = o1.state_dict();
    const float w1_next = [&] {
        Tensor loss = fn::sum(fn::mul(w1, w1));
        w1.zero_grad();
        axon::backward(loss);
        o1.step();
        return w1.at({0});
    }();

    // Reconstruct: same w after 3 steps, load state, take 1 step -> same value.
    Tensor w2 = make();
    optim::Adam o2({w2}, 0.1f);
    for (int i = 0; i < 3; ++i) {
        Tensor loss = fn::sum(fn::mul(w2, w2));
        w2.zero_grad();
        axon::backward(loss);
        o2.step();
    }
    o2.load_state_dict(state);
    Tensor loss = fn::sum(fn::mul(w2, w2));
    w2.zero_grad();
    axon::backward(loss);
    o2.step();
    EXPECT_NEAR(w2.at({0}), w1_next, 1e-5f);
}

TEST(Extra, FusedBiasActivationMatchesSeparate) {
    Tensor x = Tensor::randn({5, 4}, 1);
    Tensor bias = Tensor::from_data({4}, {0.1f, -0.2f, 0.3f, -0.4f});
    // fused == separate activation(add(x, bias))
    Tensor fused_r = ops::bias_relu(x, bias);
    Tensor sep_r = ops::relu(ops::add(x, bias));
    Tensor fused_g = ops::bias_gelu(x, bias);
    Tensor sep_g = ops::gelu(ops::add(x, bias));
    for (std::int64_t i = 0; i < x.numel(); ++i) {
        EXPECT_NEAR(fused_r.data_ptr()[i], sep_r.data_ptr()[i], 1e-5f);
        EXPECT_NEAR(fused_g.data_ptr()[i], sep_g.data_ptr()[i], 1e-5f);
    }
}

TEST(Extra, QuantizationInt8Approximates) {
    Tensor t = Tensor::from_data({4}, {-1.0f, 0.5f, 0.25f, 1.0f});
    float scale = 0.0f;
    Tensor q = ops::quantize_int8(t, scale);
    // quantized values are integers in the range [-127,127]
    for (std::int64_t i = 0; i < t.numel(); ++i) {
        EXPECT_LE(std::fabs(q.data_ptr()[i]), 127.0f);
        EXPECT_FLOAT_EQ(q.data_ptr()[i], std::round(q.data_ptr()[i]));
    }
    // dequantizing recovers the value with error <= scale
    Tensor dq = ops::dequantize_int8(q, scale);
    for (std::int64_t i = 0; i < t.numel(); ++i) {
        EXPECT_LE(std::fabs(dq.data_ptr()[i] - t.data_ptr()[i]), scale + 1e-6f);
    }
}

TEST(Extra, NumericalStabilityNoNaN) {
    // Huge logits must not produce NaN/Inf in softmax/log_softmax/cross_entropy.
    Tensor big = Tensor::from_data({2, 3}, {1000, -1000, 0, -1000, 1000, 0});
    Tensor sm = ops::softmax(big);
    Tensor ls = ops::log_softmax(big);
    for (std::int64_t i = 0; i < sm.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(sm.data_ptr()[i]));
        EXPECT_TRUE(std::isfinite(ls.data_ptr()[i]));
    }
    Tensor tgt = Tensor::from_data({2}, {0, 1});
    EXPECT_TRUE(std::isfinite(ops::cross_entropy(big, tgt).item()));
}
