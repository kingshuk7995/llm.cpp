#pragma once

#include <gpt2lab/core/tensor.hpp>
#include <gpt2lab/model/gpt2_config.hpp>

namespace gpt2lab {

class MultiHeadAttention {
public:
  MultiHeadAttention(const Gpt2Config& config);
  Tensor forward(const Tensor &x);

  Tensor w_q, b_q;
  Tensor w_k, b_k;
  Tensor w_v, b_v;
  Tensor c_proj_w, c_proj_b;
};

} // namespace gpt2lab
