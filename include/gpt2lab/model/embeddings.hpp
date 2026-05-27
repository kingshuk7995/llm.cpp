#pragma once

#include <gpt2lab/core/tensor.hpp>
#include <gpt2lab/model/gpt2_config.hpp>

namespace gpt2lab {

class Embeddings {
public:
  Embeddings(const Gpt2Config& config);
  Tensor forward(const Tensor &tokens);

  Tensor wte;
  Tensor wpe;
};

} // namespace gpt2lab
