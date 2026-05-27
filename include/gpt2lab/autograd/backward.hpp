#pragma once

#include <gpt2lab/core/tensor.hpp>

namespace gpt2lab {

Tensor backward(Tensor &loss);

} // namespace gpt2lab
