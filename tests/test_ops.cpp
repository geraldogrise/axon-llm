#include <gtest/gtest.h>

#include <array>
#include <vector>

#include "axon/ops.h"
#include "axon/tensor.h"

using axon::Tensor;
namespace ops = axon::ops;

TEST(Ops, AddElementwise) {
    Tensor a = Tensor::from_data({2, 2}, {1, 2, 3, 4});
    Tensor b = Tensor::from_data({2, 2}, {10, 20, 30, 40});
    Tensor c = ops::add(a, b);
    EXPECT_FLOAT_EQ((c.at({0, 0})), 11.0f);
    EXPECT_FLOAT_EQ((c.at({1, 1})), 44.0f);
}

TEST(Ops, SubAndMul) {
    Tensor a = Tensor::from_data({3}, {5, 6, 7});
    Tensor b = Tensor::from_data({3}, {1, 2, 3});
    EXPECT_FLOAT_EQ((ops::sub(a, b).at({0})), 4.0f);
    EXPECT_FLOAT_EQ((ops::mul(a, b).at({2})), 21.0f);
}

TEST(Ops, AddRejectsDifferentShapes) {
    Tensor a = Tensor::zeros({2, 2});
    Tensor b = Tensor::zeros({2, 3});
    EXPECT_THROW(ops::add(a, b), std::invalid_argument);
}

TEST(Ops, MatmulKnownValues) {
    // [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]
    Tensor a = Tensor::from_data({2, 2}, {1, 2, 3, 4});
    Tensor b = Tensor::from_data({2, 2}, {5, 6, 7, 8});
    Tensor c = ops::matmul(a, b);
    EXPECT_FLOAT_EQ((c.at({0, 0})), 19.0f);
    EXPECT_FLOAT_EQ((c.at({0, 1})), 22.0f);
    EXPECT_FLOAT_EQ((c.at({1, 0})), 43.0f);
    EXPECT_FLOAT_EQ((c.at({1, 1})), 50.0f);
}

TEST(Ops, MatmulNonSquare) {
    // (2x3) x (3x2) -> (2x2)
    Tensor a = Tensor::from_data({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b = Tensor::from_data({3, 2}, {7, 8, 9, 10, 11, 12});
    Tensor c = ops::matmul(a, b);
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 2);
    EXPECT_FLOAT_EQ((c.at({0, 0})), 58.0f);   // 1*7+2*9+3*11
    EXPECT_FLOAT_EQ((c.at({1, 1})), 154.0f);  // 4*8+5*10+6*12
}

TEST(Ops, MatmulWithTransposeView) {
    // Ensures matmul handles non-contiguous input (via internal contiguous).
    Tensor a = Tensor::from_data({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor at = a.transpose(0, 1);  // (3x2), non-contiguous
    Tensor b = Tensor::from_data({2, 2}, {1, 0, 0, 1});
    Tensor c = ops::matmul(at, b);  // (3x2) x (2x2) -> (3x2)
    EXPECT_FLOAT_EQ((c.at({0, 0})), 1.0f);
    EXPECT_FLOAT_EQ((c.at({0, 1})), 4.0f);
    EXPECT_FLOAT_EQ((c.at({2, 1})), 6.0f);
}

TEST(Ops, MatmulIncompatibleDimensions) {
    Tensor a = Tensor::zeros({2, 3});
    Tensor b = Tensor::zeros({2, 2});
    EXPECT_THROW((void)ops::matmul(a, b), std::invalid_argument);
}

// All optimized matmul variants must match the naive one (oracle),
// including sizes that are not multiples of 8 (exercises the AVX2 remainder).
TEST(Ops, MatmulVariantsMatchNaive) {
    for (auto dims : std::vector<std::array<std::int64_t, 3>>{{1, 1, 1}, {3, 5, 7}, {33, 17, 41}}) {
        const auto [m, k, n] = dims;
        Tensor a = Tensor::randn({m, k}, /*seed=*/11);
        Tensor b = Tensor::randn({k, n}, /*seed=*/22);
        Tensor ref = ops::detail::matmul_naive(a, b);

        for (Tensor got : {ops::detail::matmul_ikj(a, b), ops::detail::matmul_blocked(a, b),
                           ops::detail::matmul_omp(a, b), ops::detail::matmul_avx2(a, b),
                           ops::matmul(a, b)}) {
            ASSERT_EQ(got.numel(), ref.numel());
            for (std::int64_t i = 0; i < ref.numel(); ++i) {
                EXPECT_NEAR(got.data_ptr()[i], ref.data_ptr()[i], 1e-3f)
                    << "m=" << m << " k=" << k << " n=" << n << " idx=" << i;
            }
        }
    }
}
