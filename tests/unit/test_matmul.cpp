#include <gtest/gtest.h>

#include <gpt2lab/core/tensor.hpp>
#include <gpt2lab/ops/matmul.hpp>

using namespace gpt2lab;

TEST(MatmulTest, OutputShape) {
  Tensor a({2, 3});
  Tensor b({3, 2});

  Tensor c = matmul(a, b);

  EXPECT_EQ(c.shape().dim(0), 2);
  EXPECT_EQ(c.shape().dim(1), 2);
}