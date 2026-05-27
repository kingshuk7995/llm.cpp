#pragma once

#include <gpt2lab/core/tensor.hpp>
#include <gpt2lab/model/gpt2_config.hpp>

namespace gpt2lab {

class MLP {
public:
  MLP(const Gpt2Config& config);
  Tensor forward(const Tensor &x);

  Tensor c_fc_w, c_fc_b;
  Tensor c_proj_w, c_proj_b;
};

} // namespace gpt2lab
