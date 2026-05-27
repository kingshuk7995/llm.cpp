#pragma once

#include <gpt2lab/core/tensor.hpp>

namespace gpt2lab {

Tensor multi_head_attention(const Tensor &x);

} // namespace gpt2lab
