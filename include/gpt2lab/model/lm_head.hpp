#pragma once

#include <gpt2lab/core/tensor.hpp>

namespace gpt2lab {

Tensor lm_head(const Tensor &hidden);

} // namespace gpt2lab
