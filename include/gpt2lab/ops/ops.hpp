#pragma once

#include <gpt2lab/core/tensor.hpp>

namespace gpt2lab {

Tensor add(const Tensor &a, const Tensor &b);
Tensor matmul(const Tensor &a, const Tensor &b);
Tensor batched_matmul(const Tensor &a, const Tensor &b, bool trans_b = false);
Tensor gelu(const Tensor &x);
Tensor layernorm(const Tensor &x, const Tensor &gamma, const Tensor &beta, float eps = 1e-5f);
Tensor softmax(const Tensor &x, int dim = -1);

} // namespace gpt2lab
