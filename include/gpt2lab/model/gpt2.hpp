#pragma once

#include <gpt2lab/core/tensor.hpp>

#include <gpt2lab/model/gpt2_config.hpp>

namespace gpt2lab {

class Gpt2Model {
public:
  explicit Gpt2Model(const Gpt2Config &config);
  Tensor forward(const Tensor &input);

private:
  Gpt2Config config_;
};

} // namespace gpt2lab
