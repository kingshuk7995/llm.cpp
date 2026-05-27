#include <gtest/gtest.h>

#include <gpt2lab/backends/cuda_backend.hpp>
#include <gpt2lab/core/tensor.hpp>

using namespace gpt2lab;

TEST(CudaTest, MatmulExecution) {
  CudaBackend backend;

  Tensor a({2, 2});
  a.data()[0] = 1.0f; a.data()[1] = 2.0f;
  a.data()[2] = 3.0f; a.data()[3] = 4.0f;

  Tensor b({2, 2});
  b.data()[0] = 2.0f; b.data()[1] = 0.0f;
  b.data()[2] = 1.0f; b.data()[3] = 2.0f;

  // Expected output:
  // [1 2] * [2 0] = [4 4]
  // [3 4]   [1 2]   [10 8]
  
  Tensor c = backend.matmul(a, b);

  EXPECT_EQ(c.shape().dim(0), 2);
  EXPECT_EQ(c.shape().dim(1), 2);

  EXPECT_FLOAT_EQ(c.data()[0], 4.0f);
  EXPECT_FLOAT_EQ(c.data()[1], 4.0f);
  EXPECT_FLOAT_EQ(c.data()[2], 10.0f);
  EXPECT_FLOAT_EQ(c.data()[3], 8.0f);
}
