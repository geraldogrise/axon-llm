#include <gtest/gtest.h>

#include <memory>

#include "axon/autograd.h"
#include "axon/functional.h"
#include "axon/nn.h"
#include "axon/ops.h"
#include "axon/optim.h"

using namespace axon;

TEST(Recurrent, RNNOutputShape) {
    nn::RNN rnn(4, 8, 1);
    Tensor x = Tensor::randn({5, 4}, 2);  // L=5, in=4
    Tensor y = rnn.forward(x);
    EXPECT_EQ(y.shape()[0], 5);
    EXPECT_EQ(y.shape()[1], 8);
    EXPECT_EQ(rnn.parameters().size(), 4u);  // 2 Linear x (W,b)
}

TEST(Recurrent, LSTMOutputShape) {
    nn::LSTM lstm(4, 6, 1);
    Tensor x = Tensor::randn({3, 4}, 2);
    Tensor y = lstm.forward(x);
    EXPECT_EQ(y.shape()[0], 3);
    EXPECT_EQ(y.shape()[1], 6);
}

// The LSTM should learn to predict the next element of a sequence (overfit).
TEST(Recurrent, LSTMTrainsSequence) {
    // Input: one-hot of 5 tokens in [0..4]; target: next token.
    nn::LSTM lstm(5, 16, 3);
    nn::Linear head(16, 5, 4);
    Tensor x = Tensor::from_data({4, 5}, {1, 0, 0, 0, 0,   0, 1, 0, 0, 0,
                                          0, 0, 1, 0, 0,   0, 0, 0, 1, 0});
    Tensor tgt = Tensor::from_data({4}, {1, 2, 3, 4});

    std::vector<Tensor> params = lstm.parameters();
    for (Tensor& p : head.parameters()) params.push_back(p);
    optim::Adam opt(params, 0.02f);

    float first = 0.0f, last = 0.0f;
    for (int e = 0; e < 300; ++e) {
        Tensor logits = head.forward(lstm.forward(x));
        Tensor loss = fn::cross_entropy(logits, tgt);
        if (e == 0) first = loss.item();
        last = loss.item();
        opt.zero_grad();
        backward(loss);
        opt.step();
    }
    EXPECT_LT(last, first * 0.3f);
}

TEST(NNExtra, DropoutTrainVsInference) {
    nn::Dropout drop(0.5f, 7);
    Tensor x = Tensor::ones({100, 10});
    // Training (grad on): some elements zeroed, others scaled by 2.
    Tensor y_train = drop.forward(x);
    int zeros = 0;
    for (std::int64_t i = 0; i < y_train.numel(); ++i) {
        if (y_train.data_ptr()[i] == 0.0f) ++zeros;
    }
    EXPECT_GT(zeros, 0);  // dropped something
    // Inference (no_grad): identity.
    NoGradGuard g;
    Tensor y_eval = drop.forward(x);
    EXPECT_FLOAT_EQ(y_eval.at({0, 0}), 1.0f);
}

TEST(NNExtra, AdamWDecaysWeights) {
    // With zero grad and weight_decay>0, weights should shrink toward zero.
    Tensor w = Tensor::from_data({3}, {1.0f, -2.0f, 3.0f});
    w.requires_grad_(true);
    // produce a null grad (backward of 0*sum) so w has has_grad
    Tensor loss = fn::mul(fn::sum(w), Tensor::zeros({1}));
    backward(loss);
    optim::Adam opt({w}, /*lr=*/0.1f, 0.9f, 0.999f, 1e-8f, /*weight_decay=*/0.1f);
    const float before = std::abs(w.at({0}));
    opt.step();
    EXPECT_LT(std::abs(w.at({0})), before);  // shrank
}

TEST(Matmul, VariantAVX512MatchesNaive) {
    Tensor a = Tensor::randn({33, 17}, 1);
    Tensor b = Tensor::randn({17, 41}, 2);
    Tensor ref = ops::detail::matmul_naive(a, b);
    Tensor got = ops::detail::matmul_avx512(a, b);  // falls back to OMP if no AVX-512
    for (std::int64_t i = 0; i < ref.numel(); ++i) {
        EXPECT_NEAR(got.data_ptr()[i], ref.data_ptr()[i], 1e-3f);
    }
}
