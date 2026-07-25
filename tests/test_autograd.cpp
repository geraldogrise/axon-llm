#include <gtest/gtest.h>

#include <functional>

#include "axon/autograd.h"
#include "axon/functional.h"
#include "axon/ops.h"

using axon::Tensor;
namespace fn = axon::fn;

namespace {

// Numerical gradient check via finite differences:
// dL/dx[i] ~= (L(x+h) - L(x-h)) / (2h). Compared against the autograd grad.
void check_grad(const std::function<Tensor(Tensor&)>& build_loss, Tensor& x, float tol = 1e-2f) {
    // Analytic gradient via autograd.
    Tensor loss = build_loss(x);
    axon::backward(loss);
    Tensor analytic = x.grad().clone();

    // Numerical gradient.
    const float h = 1e-3f;
    const std::int64_t n = x.numel();
    for (std::int64_t i = 0; i < n; ++i) {
        const float orig = x.data_ptr()[i];

        x.data_ptr()[i] = orig + h;
        const float lp = build_loss(x).item();
        x.data_ptr()[i] = orig - h;
        const float lm = build_loss(x).item();
        x.data_ptr()[i] = orig;

        const float num = (lp - lm) / (2.0f * h);
        EXPECT_NEAR(analytic.data_ptr()[i], num, tol) << "at index " << i;
    }
}

}  // namespace

TEST(Autograd, SumAndMeanGradient) {
    Tensor x = Tensor::from_data({4}, {1, 2, 3, 4});
    x.requires_grad_(true);
    Tensor loss = fn::sum(x);
    axon::backward(loss);
    // d(sum)/dx = 1 for every element.
    for (std::int64_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(x.grad().data_ptr()[i], 1.0f);
    }
}

TEST(Autograd, MulAccumulatesWhenInputUsedTwice) {
    // f = sum(x * x) => df/dx = 2x
    Tensor x = Tensor::from_data({3}, {1, 2, 3});
    x.requires_grad_(true);
    Tensor loss = fn::sum(fn::mul(x, x));
    axon::backward(loss);
    EXPECT_FLOAT_EQ(x.grad().data_ptr()[0], 2.0f);
    EXPECT_FLOAT_EQ(x.grad().data_ptr()[1], 4.0f);
    EXPECT_FLOAT_EQ(x.grad().data_ptr()[2], 6.0f);
}

TEST(Autograd, GradCheckMatmulRelu) {
    Tensor w = Tensor::randn({3, 2}, /*seed=*/7);
    w.requires_grad_(true);
    Tensor input = Tensor::from_data({1, 3}, {0.5f, -1.0f, 2.0f});
    Tensor target = Tensor::from_data({1, 2}, {1.0f, 0.0f});

    check_grad([&](Tensor& p) { return fn::mse_loss(fn::relu(fn::matmul(input, p)), target); }, w);
}

TEST(Autograd, GradCheckAddBiasBroadcast) {
    // Bias added by broadcasting: grad must sum over the batch axis.
    Tensor b = Tensor::from_data({2}, {0.1f, -0.2f});
    b.requires_grad_(true);
    Tensor input = Tensor::from_data({3, 2}, {1, 2, 3, 4, 5, 6});
    Tensor target = Tensor::zeros({3, 2});

    check_grad([&](Tensor& p) { return fn::mse_loss(fn::add(input, p), target); }, b);
}

TEST(Autograd, ZeroGradClears) {
    Tensor x = Tensor::from_data({2}, {1, 2});
    x.requires_grad_(true);
    Tensor loss = fn::sum(x);
    axon::backward(loss);
    EXPECT_TRUE(x.has_grad());
    x.zero_grad();
    EXPECT_FALSE(x.has_grad());
}
