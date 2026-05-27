#pragma once

#include <gpt2lab/core/tensor.hpp>

#include <gpt2lab/model/gpt2_config.hpp>
#include <gpt2lab/model/embeddings.hpp>
#include <gpt2lab/model/transformer_block.hpp>
#include <vector>

namespace gpt2lab {

class Gpt2Model {
public:
  explicit Gpt2Model(const Gpt2Config &config);
  Tensor forward(const Tensor &input);

private:
  Gpt2Config config_;
  Embeddings embeddings_;
  std::vector<TransformerBlock> blocks_;
  Tensor lm_head_w;
};

} // namespace gpt2lab
