#include <gtest/gtest.h>

#include <memory>

#include "axon/autograd.h"
#include "axon/functional.h"
#include "axon/nn.h"
#include "axon/optim.h"

using namespace axon;

TEST(NN, LinearOutputShape) {
    nn::Linear fc(3, 4);
    Tensor x = Tensor::zeros({2, 3});  // batch=2, in=3
    Tensor y = fc.forward(x);
    ASSERT_EQ(y.ndim(), 2);
    EXPECT_EQ(y.shape()[0], 2);
    EXPECT_EQ(y.shape()[1], 4);
}

TEST(NN, LinearHasTwoParameters) {
    nn::Linear fc(3, 4);
    auto params = fc.parameters();
    ASSERT_EQ(params.size(), 2u);        // weight + bias
    EXPECT_EQ(params[0].numel(), 12);    // 3*4
    EXPECT_EQ(params[1].numel(), 4);     // bias
}

// Integration test: the network actually learns XOR (loss drops a lot and the
// predictions land on the correct side of 0.5).
TEST(NN, TrainsXOR) {
    Tensor x = Tensor::from_data({4, 2}, {0, 0, 0, 1, 1, 0, 1, 1});
    Tensor y = Tensor::from_data({4, 1}, {0, 1, 1, 0});

    auto model = std::make_shared<nn::Sequential>();
    model->add(std::make_shared<nn::Linear>(2, 8, 1));
    model->add(std::make_shared<nn::ReLU>());
    model->add(std::make_shared<nn::Linear>(8, 1, 2));

    optim::Adam opt(model->parameters(), 0.05f);

    float first_loss = 0.0f;
    float last_loss = 0.0f;
    for (int epoch = 0; epoch < 2000; ++epoch) {
        Tensor pred = model->forward(x);
        Tensor loss = fn::mse_loss(pred, y);
        if (epoch == 0) {
            first_loss = loss.item();
        }
        last_loss = loss.item();
        opt.zero_grad();
        backward(loss);
        opt.step();
    }

    EXPECT_LT(last_loss, first_loss * 0.1f);  // loss dropped >= 10x
    EXPECT_LT(last_loss, 0.05f);              // actually converged

    Tensor pred = model->forward(x);
    EXPECT_LT(pred.at({0, 0}), 0.5f);  // 0 XOR 0 = 0
    EXPECT_GT(pred.at({1, 0}), 0.5f);  // 0 XOR 1 = 1
    EXPECT_GT(pred.at({2, 0}), 0.5f);  // 1 XOR 0 = 1
    EXPECT_LT(pred.at({3, 0}), 0.5f);  // 1 XOR 1 = 0
}
