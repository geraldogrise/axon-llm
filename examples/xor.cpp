// Example: train a network to learn the XOR function.
// XOR is not linearly separable, so it requires a hidden layer + activation.
// Demonstrates: Tensor, autograd, nn.Linear/ReLU/Sequential and the Adam optimizer.

#include <cstdio>
#include <memory>

#include "axon/autograd.h"
#include "axon/functional.h"
#include "axon/nn.h"
#include "axon/optim.h"

using namespace axon;

int main() {
    // XOR data: 4 examples, 2 inputs, 1 output.
    Tensor x = Tensor::from_data({4, 2}, {0, 0, 0, 1, 1, 0, 1, 1});
    Tensor y = Tensor::from_data({4, 1}, {0, 1, 1, 0});

    // Model: 2 -> 8 -> ReLU -> 1
    auto model = std::make_shared<nn::Sequential>();
    model->add(std::make_shared<nn::Linear>(2, 8, /*seed=*/1));
    model->add(std::make_shared<nn::ReLU>());
    model->add(std::make_shared<nn::Linear>(8, 1, /*seed=*/2));

    optim::Adam opt(model->parameters(), /*lr=*/0.05f);

    std::printf("Training XOR...\n");
    for (int epoch = 1; epoch <= 2000; ++epoch) {
        Tensor pred = model->forward(x);
        Tensor loss = fn::mse_loss(pred, y);

        opt.zero_grad();
        backward(loss);
        opt.step();

        if (epoch % 200 == 0 || epoch == 1) {
            std::printf("epoch %4d | loss = %.6f\n", epoch, loss.item());
        }
    }

    // Final predictions.
    Tensor pred = model->forward(x);
    std::printf("\nResult (expected: 0, 1, 1, 0):\n");
    for (std::int64_t i = 0; i < 4; ++i) {
        std::printf("  [%.0f, %.0f] -> %.3f\n", x.at({i, 0}), x.at({i, 1}), pred.at({i, 0}));
    }
    return 0;
}
