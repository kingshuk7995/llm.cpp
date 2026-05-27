#pragma once

#include <gpt2lab/core/tensor.hpp>

namespace gpt2lab {

class TransformerBlock {
public:
  Tensor operator()(const Tensor &x) const;
};

} // namespace gpt2lab
