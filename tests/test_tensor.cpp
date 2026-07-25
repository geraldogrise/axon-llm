#include <gtest/gtest.h>

#include "axon/tensor.h"

using axon::Tensor;

TEST(Tensor, ZerosHasCorrectShapeAndValues) {
    Tensor t = Tensor::zeros({2, 3});
    EXPECT_EQ(t.ndim(), 2);
    EXPECT_EQ(t.numel(), 6);
    EXPECT_TRUE(t.is_contiguous());
    for (std::int64_t i = 0; i < 2; ++i) {
        for (std::int64_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ((t.at({i, j})), 0.0f);
        }
    }
}

TEST(Tensor, OnesAndFull) {
    EXPECT_FLOAT_EQ((Tensor::ones({4}).at({2})), 1.0f);
    EXPECT_FLOAT_EQ((Tensor::full({2, 2}, 3.5f).at({1, 1})), 3.5f);
}

TEST(Tensor, FromDataPreservesRowMajorOrder) {
    Tensor t = Tensor::from_data({2, 2}, {1, 2, 3, 4});
    EXPECT_FLOAT_EQ((t.at({0, 0})), 1.0f);
    EXPECT_FLOAT_EQ((t.at({0, 1})), 2.0f);
    EXPECT_FLOAT_EQ((t.at({1, 0})), 3.0f);
    EXPECT_FLOAT_EQ((t.at({1, 1})), 4.0f);
}

TEST(Tensor, FromDataRejectsWrongSize) {
    EXPECT_THROW(Tensor::from_data({2, 2}, {1, 2, 3}), std::invalid_argument);
}

TEST(Tensor, TransposeIsView) {
    Tensor t = Tensor::from_data({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor tt = t.transpose(0, 1);
    EXPECT_EQ(tt.shape()[0], 3);
    EXPECT_EQ(tt.shape()[1], 2);
    EXPECT_FALSE(tt.is_contiguous());
    // t[i][j] == tt[j][i]
    EXPECT_FLOAT_EQ((tt.at({0, 1})), (t.at({1, 0})));
    EXPECT_FLOAT_EQ((tt.at({2, 1})), (t.at({1, 2})));
}

TEST(Tensor, ContiguousMaterializesTranspose) {
    Tensor t = Tensor::from_data({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor c = t.transpose(0, 1).contiguous();
    EXPECT_TRUE(c.is_contiguous());
    // Expected layout after transposing: [1,4, 2,5, 3,6]
    const float* p = c.data_ptr();
    EXPECT_FLOAT_EQ(p[0], 1.0f);
    EXPECT_FLOAT_EQ(p[1], 4.0f);
    EXPECT_FLOAT_EQ(p[2], 2.0f);
    EXPECT_FLOAT_EQ(p[3], 5.0f);
}

TEST(Tensor, ReshapeSharesData) {
    Tensor t = Tensor::from_data({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor r = t.reshape({3, 2});
    EXPECT_EQ(r.shape()[0], 3);
    EXPECT_EQ(r.shape()[1], 2);
    EXPECT_FLOAT_EQ((r.at({0, 0})), 1.0f);
    EXPECT_FLOAT_EQ((r.at({2, 1})), 6.0f);
}

TEST(Tensor, ReshapeRejectsIncompatibleNumel) {
    Tensor t = Tensor::zeros({2, 3});
    EXPECT_THROW((void)t.reshape({2, 2}), std::invalid_argument);
}

TEST(Tensor, AtOutOfBoundsThrows) {
    Tensor t = Tensor::zeros({2, 2});
    EXPECT_THROW((void)t.at({2, 0}), std::out_of_range);
}
