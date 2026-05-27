#pragma once

#include <gpt2lab/core/tensor.hpp>
#include <gpt2lab/model/gpt2_config.hpp>
#include <gpt2lab/model/mha.hpp>
#include <gpt2lab/model/mlp.hpp>

namespace gpt2lab {

class TransformerBlock {
public:
  TransformerBlock(const Gpt2Config& config);
  Tensor forward(const Tensor &x);

  MultiHeadAttention attn;
  MLP mlp;
};

} // namespace gpt2lab
