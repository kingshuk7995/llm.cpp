#include <gtest/gtest.h>

#include <gpt2lab/autograd/backward.hpp>
#include <gpt2lab/core/tensor.hpp>
#include <gpt2lab/ops/ops.hpp>

using namespace gpt2lab;

TEST(AutogradTest, BackwardShape) {
  Tensor loss({1, 1});
  loss.set_requires_grad(true);

  Tensor grad = backward(loss);

  EXPECT_EQ(grad.shape().dim(0), 1);
  EXPECT_EQ(grad.shape().dim(1), 1);
}

TEST(AutogradTest, AddAccumulation) {
  Tensor a({2});
  a.fill(1.0f);
  a.set_requires_grad(true);

  Tensor b({2});
  b.fill(2.0f);
  b.set_requires_grad(true);

  // out = a + b
  Tensor out = gpt2lab::add(a, b);
  
  // out.grad = 1.0
  backward(out);

  EXPECT_FLOAT_EQ(a.grad().data()[0], 1.0f);
  EXPECT_FLOAT_EQ(a.grad().data()[1], 1.0f);
  EXPECT_FLOAT_EQ(b.grad().data()[0], 1.0f);
  EXPECT_FLOAT_EQ(b.grad().data()[1], 1.0f);
}