#pragma once
#include <gpt2lab/core/tensor.hpp>

namespace gpt2lab {

void init_normal(Tensor& tensor, float mean, float stddev);
void init_zeros(Tensor& tensor);

} // namespace gpt2lab
