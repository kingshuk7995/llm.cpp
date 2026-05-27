#include <gpt2lab/model/gpt2.hpp>
#include <gpt2lab/model/embeddings.hpp>
#include <gpt2lab/model/lm_head.hpp>

namespace gpt2lab {

Gpt2Model::Gpt2Model(const Gpt2Config &config) : config_(config) {}

Tensor Gpt2Model::forward(const Tensor &input) {
  Tensor hidden = embed_tokens(input);
  return lm_head(hidden);
}

} // namespace gpt2lab
