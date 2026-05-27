#include <gpt2lab/model/gpt2.hpp>
#include <gpt2lab/model/embeddings.hpp>
#include <gpt2lab/model/gpt2.hpp>
#include <gpt2lab/model/init.hpp>
#include <gpt2lab/ops/layernorm.hpp>
#include <gpt2lab/ops/matmul.hpp>

namespace gpt2lab {

Gpt2Model::Gpt2Model(const Gpt2Config &config)
    : config_(config), embeddings_(config) {
  
  for (int i = 0; i < config.num_layers; ++i) {
    blocks_.emplace_back(config);
  }

  lm_head_w = Tensor({(size_t)config.hidden_size, (size_t)config.vocab_size}, DType::Float32);
  init_normal(lm_head_w, 0.0f, 0.02f);
}

Tensor Gpt2Model::forward(const Tensor &input) {
  Tensor x = embeddings_.forward(input);

  for (auto& block : blocks_) {
    x = block.forward(x);
  }

  x = layernorm(x);
  Tensor logits = matmul(x, lm_head_w);

  return logits;
}

} // namespace gpt2lab
