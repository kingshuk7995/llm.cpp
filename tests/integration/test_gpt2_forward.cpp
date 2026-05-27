#include <gtest/gtest.h>

#include <gpt2lab/core/tensor.hpp>
#include <gpt2lab/model/gpt2.hpp>

using namespace gpt2lab;

TEST(Gpt2IntegrationTest, ForwardPass) {
  Gpt2Config config;
  Gpt2Model model(config);
  Tensor input({1, 1});
  Tensor output = model.forward(input);
  EXPECT_EQ(output.shape().dim(0), 1);
  EXPECT_EQ(output.shape().dim(1), 50257);
}
