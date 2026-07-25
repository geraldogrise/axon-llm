#include <gtest/gtest.h>

#include <cmath>
#include <functional>

#include "axon/autograd.h"
#include "axon/functional.h"
#include "axon/ops.h"

using axon::Tensor;
namespace fn = axon::fn;
namespace ops = axon::ops;

namespace {

// Gradient check: compares the autograd grad with finite differences.
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
        const float num = (lp - lm) / (2.0f * h);
        EXPECT_NEAR(analytic.data_ptr()[i], num, tol) << "index " << i;
    }
}

}  // namespace

TEST(Activations, SigmoidKnownValues) {
    Tensor x = Tensor::from_data({3}, {0.0f, 100.0f, -100.0f});
    Tensor y = ops::sigmoid(x);
    EXPECT_NEAR(y.data_ptr()[0], 0.5f, 1e-6f);
    EXPECT_NEAR(y.data_ptr()[1], 1.0f, 1e-6f);   // stable, no overflow
    EXPECT_NEAR(y.data_ptr()[2], 0.0f, 1e-6f);   // stable, no overflow
}

TEST(Activations, GradCheckSigmoid) {
    Tensor x = Tensor::from_data({4}, {-2, -0.5f, 0.5f, 2});
    x.requires_grad_(true);
    check_grad([](Tensor& p) { return fn::sum(fn::sigmoid(p)); }, x);
}

TEST(Activations, GradCheckTanh) {
    Tensor x = Tensor::from_data({4}, {-2, -0.5f, 0.5f, 2});
    x.requires_grad_(true);
    check_grad([](Tensor& p) { return fn::sum(fn::tanh(p)); }, x);
}

TEST(Activations, GradCheckGelu) {
    Tensor x = Tensor::from_data({5}, {-2, -0.5f, 0.0f, 0.5f, 2});
    x.requires_grad_(true);
    check_grad([](Tensor& p) { return fn::sum(fn::gelu(p)); }, x);
}

TEST(Activations, SoftmaxSumsToOne) {
    Tensor x = Tensor::from_data({2, 3}, {1, 2, 3, 1, 1, 1});
    Tensor y = ops::softmax(x);
    for (std::int64_t i = 0; i < 2; ++i) {
        float s = 0.0f;
        for (std::int64_t j = 0; j < 3; ++j) {
            s += y.at({i, j});
        }
        EXPECT_NEAR(s, 1.0f, 1e-6f);
    }
    // Uniform row -> equal probabilities.
    EXPECT_NEAR(y.at({1, 0}), 1.0f / 3.0f, 1e-6f);
}

TEST(Activations, GradCheckSoftmax) {
    // Loss = sum(softmax(x) * weight), with fixed weights, to produce a non-trivial gradient.
    Tensor x = Tensor::from_data({2, 3}, {0.1f, 0.5f, -0.3f, 1.0f, -1.0f, 0.2f});
    x.requires_grad_(true);
    Tensor w = Tensor::from_data({2, 3}, {1, 2, 3, 4, 5, 6});
    check_grad([&](Tensor& p) { return fn::sum(fn::mul(fn::softmax(p), w)); }, x);
}

TEST(Activations, CrossEntropyValueAndGrad) {
    // Perfect logits for the correct class -> low loss.
    Tensor logits = Tensor::from_data({2, 3}, {10, 0, 0, 0, 10, 0});
    Tensor targets = Tensor::from_data({2}, {0, 1});
    Tensor loss = ops::cross_entropy(logits, targets);
    EXPECT_LT(loss.item(), 0.01f);

    // Gradient check of cross-entropy with respect to the logits.
    Tensor x = Tensor::from_data({2, 3}, {0.5f, -0.2f, 0.1f, -0.3f, 0.8f, 0.0f});
    x.requires_grad_(true);
    Tensor tgt = Tensor::from_data({2}, {2, 0});
    check_grad([&](Tensor& p) { return fn::cross_entropy(p, tgt); }, x);
}
