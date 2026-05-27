#pragma once

#include <gpt2lab/core/tensor.hpp>

namespace gpt2lab {

Tensor add(const Tensor &a, const Tensor &b);
Tensor matmul(const Tensor &a, const Tensor &b);
Tensor gelu(const Tensor &x);
Tensor layernorm(const Tensor &x);
Tensor softmax(const Tensor &x, int dim = -1);

} // namespace gpt2lab
