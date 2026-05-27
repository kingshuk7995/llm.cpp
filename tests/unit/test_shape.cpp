#include <gtest/gtest.h>

#include <gpt2lab/core/shape.hpp>

using namespace gpt2lab;

TEST(ShapeTest, Construction) {
  Shape s{2, 3};

  EXPECT_EQ(s.dim(0), 2);
  EXPECT_EQ(s.dim(1), 3);
  EXPECT_EQ(shape_size(s), 6);
}