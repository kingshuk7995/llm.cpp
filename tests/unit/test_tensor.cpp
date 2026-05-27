#include <gtest/gtest.h>

#include <gpt2lab/core/tensor.hpp>

using namespace gpt2lab;

TEST(TensorTest, FillAndSize) {
  Tensor t({2, 2});

  t.fill(1.0f);

  EXPECT_EQ(t.size(), 4);
}